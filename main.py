from data import *
from net import *
from lib import *
from easydl import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import os
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns
cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

if args.misc.gpus < 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.misc.gpu_id
    gpu_ids = args.misc.gpu_id_list

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

if args.train.continue_training:
    log_dir = f'{args.log.root_dir}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
}
batch_size = args.data.dataloader.batch_size

def sns_plot(para_source1, para_source2, para_target, source1_shared_index, source1_private_index, source2_shared_index
             , source2_private_index, target_shared_index, target_private_index, global_step, name, log=False, save=False):
    source1_shared = torch.index_select(para_source1, dim=0, index=source1_shared_index).flatten().cpu().detach().numpy()
    source1_private = torch.index_select(para_source1, dim=0, index=source1_private_index).flatten().cpu().detach().numpy()
    source2_shared = torch.index_select(para_source2, dim=0, index=source2_shared_index).flatten().cpu().detach().numpy()
    source2_private = torch.index_select(para_source2, dim=0, index=source2_private_index).flatten().cpu().detach().numpy()
    target_shared = torch.index_select(para_target, dim=0, index=target_shared_index).flatten().cpu().detach().numpy()
    target_private = torch.index_select(para_target, dim=0, index=target_private_index).flatten().cpu().detach().numpy()
    if log:
        logger.add_scalar('weight/source1_shared_weight', source1_shared.mean(), global_step)
        logger.add_scalar('weight/source1_private_weight', source1_private.mean(), global_step)
        logger.add_scalar('weight/source2_shared_weight', source2_shared.mean(), global_step)
        logger.add_scalar('weight/source2_private_weight', source2_private.mean(), global_step)
        logger.add_scalar('weight/target_shared_weight', target_shared.mean(), global_step)
        logger.add_scalar('weight/target_private_weight', target_private.mean(), global_step)
    if save:
        sns.set()
        sns.kdeplot(source1_shared, cut=0, label='source#1 shared')
        sns.kdeplot(source1_private, cut=0, label='source#1 private')
        sns.kdeplot(source2_shared, cut=0, label='source#2 shared')
        sns.kdeplot(source2_private, cut=0, label='source#2 private')
        sns.kdeplot(target_shared, cut=0, label='target shared')
        sns.kdeplot(target_private, cut=0, label='target private')
        plt.legend()
        plt.savefig(join(log_dir, name ))
        plt.close()


def label_to_RGB(label):
    color = np.zeros((len(label), 3))
    for index in range(len(label)):
        if label[index] == 0:
            color[index] = np.array([1, 0, 0])   # red: target samples wrong aligned to known classes
        if label[index] == 1:
            color[index] = np.array([0, 1, 0])   # green: source and target samples with shared labels
        if label[index] == 2:
            color[index] = np.array([0, 0, 1])   # blue: source samples with source private labels
        if label[index] == 3:
            color[index] = np.array([0, 0, 0])   # black: target samples with target private labels (refer as the unknown category)
    return color

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim)
        self.domain_discriminator = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        y = self.classifier(f)
        d = self.domain_discriminator(f)
        return y, d


totalNet = TotalNet()
feature_extractor = nn.DataParallel(totalNet.feature_extractor.cuda(), device_ids=gpu_ids).train(True)
classifier = nn.DataParallel(totalNet.classifier.cuda(), device_ids=gpu_ids).train(True)
domain_discriminator = nn.DataParallel(totalNet.domain_discriminator.cuda(), device_ids=gpu_ids).train(True)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    domain_discriminator.load_state_dict(data['domain_discriminator'])

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label','shared_weight']) as target_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.cuda()
            label = label.cuda()

            feature = feature_extractor.forward(im)
            feature, fc1, before_softmax, predict_prob = classifier.forward(feature)
            shared_weight = (predict_prob.max(1)[0] - torch.sort(predict_prob, dim=1, descending=True)[0][:, 1])

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['feature_source1', 'label_source1']) as target_accumulator, \
            torch.no_grad():
        for i, (im_source1, label_source1) in enumerate(tqdm(source1_test_dl, desc='testing ')):
            im_source1 = im_source1.cuda()
            label_source1 = label_source1.cuda()

            feature_source1 = feature_extractor.forward(im_source1)
            feature_source1, fc1_s1, before_softmax_s1, predict_prob_s1 = classifier.forward(feature_source1)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['feature_source2', 'label_source2']) as target_accumulator, \
            torch.no_grad():
        for i, (im_source2, label_source2) in enumerate(tqdm(source2_test_dl, desc='testing ')):
            im_source2 = im_source2.cuda()
            label_source2 = label_source2.cuda()

            feature_source2 = feature_extractor.forward(im_source2)
            feature_source2, fc1_s2, before_softmax_s2, predict_prob_s2 = classifier.forward(feature_source2)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

    for (each_predict_prob, each_label, each_shared_weight) in zip(predict_prob, label,shared_weight):
        each_pred_id = np.argmax(each_predict_prob)
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            if each_pred_id == each_label and each_shared_weight >= args.test.w_0: #
                counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            if each_shared_weight < args.test.w_0:
                counters[-1].Ncorrect += 1.0

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    print(f'test accuracy is {acc_test.item()}')

    # for i in range(args.data.dataset.n_share):
    #     logger.add_scalar(f'acc_per_class/{i}', counters[i].Ncorrect / (counters[i].Ntotal + 1e-10), 1)

    feature_list = np.concatenate((feature, feature_source1, feature_source2), axis=0)
    Y = TSNE(n_jobs=4).fit_transform(feature_list)
    plt.scatter(Y[:len(label), 0], Y[:len(label), 1], s=5, c=label, marker='s')
    plt.scatter(Y[len(label):len(label)+len(label_source1), 0], Y[len(label):len(label)+len(label_source1), 1], s=5, c=label_source1, marker='.')
    plt.scatter(Y[len(label)+len(label_source1):, 0], Y[len(label)+len(label_source1):, 1], s=5, c=label_source2, marker='^')
    plt.savefig('./log/{}/test_distribution.png'.format(now))
    plt.close()
    exit(0)

if args.train.continue_training:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    domain_discriminator.load_state_dict(data['domain_discriminator'])

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
if args.misc.gpus > 1:
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.module.parameters(), lr=args.train.lr / 10, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.module.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_domain_discriminator = OptimWithSheduler(
        optim.SGD(domain_discriminator.module.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
else:
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10, weight_decay=args.train.weight_decay,momentum=args.train.momentum, nesterov=True),scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,momentum=args.train.momentum, nesterov=True),scheduler)
    optimizer_domain_discriminator = OptimWithSheduler(
        optim.SGD(domain_discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,momentum=args.train.momentum, nesterov=True),scheduler)

global_step = 0 + args.train.continue_step
records = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step - args.train.continue_step),desc='global step')
epoch_id = 0
class_temperture = torch.zeros(len(source_classes), 1).cuda()

while global_step < args.train.min_step:

    iters = tqdm(zip(source1_train_dl, source2_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source1_train_dl), len(source2_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source1, label_source1), (im_source2, label_source2), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage
        label_source1 = torch.where(label_source1 >= args.data.dataset.n_total, label_source1 - 5, label_source1)
        label_source2 = torch.where(label_source2 >= args.data.dataset.n_total, label_source2 - 5, label_source2)
        label_source1 = Variable(label_source1.cuda())
        label_source2 = Variable(label_source2.cuda())
        label_target = Variable(label_target.cuda())
        # label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source1 = Variable(im_source1.cuda())
        im_source2 = Variable(im_source2.cuda())
        im_target = Variable(im_target.cuda())

        fc1_s1 = feature_extractor.forward(im_source1)
        fc1_s2 = feature_extractor.forward(im_source2)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s1, feature_source1, fc2_s1, predict_prob_source1 = classifier.forward(fc1_s1)
        fc1_s2, feature_source2, fc2_s2, predict_prob_source2 = classifier.forward(fc1_s2)
        fc1_t, feature_target, fc2_t, predict_prob_target= classifier.forward(fc1_t)

        domain_prob_source1 = domain_discriminator.forward(feature_source1)
        domain_prob_source2 = domain_discriminator.forward(feature_source2)
        domain_prob_target = domain_discriminator.forward(feature_target)

        with torch.no_grad():
            source1_shared_weight = torch.zeros(batch_size,1).cuda()
            source2_shared_weight = torch.zeros(batch_size, 1).cuda()
            target_shared_weight = torch.zeros(batch_size, 1).cuda()
            target_accumulated_margin = torch.zeros(len(source_classes), 1).cuda()
            target_pseudo_label = predict_prob_target.max(1)[1]
            num_per_class = torch.zeros(len(source_classes), 1).cuda()
            sorted_pred_target = torch.sort(predict_prob_target, dim=1, descending=True)[0]
            target_margin = (predict_prob_target.max(1)[0] - torch.sort(predict_prob_target, dim=1, descending=True)[0][:,1]).view(batch_size, 1)
            for index in range(batch_size):
                target_accumulated_margin[target_pseudo_label[index],0] += sorted_pred_target[index,0] - sorted_pred_target[index,1]
                num_per_class[target_pseudo_label[index],0] +=1
            target_pred_per_label = ((class_temperture * records) + torch.div(target_accumulated_margin, num_per_class + 1e-6)) / (records + 1)
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source1, len(source_classes))), variable_to_numpy(predict_prob_source1))
            counter.addOneBatch(variable_to_numpy(one_hot(label_source2, len(source_classes))), variable_to_numpy(predict_prob_source2))
            acc_train = torch.tensor([counter.reportAccuracy()]).cuda()
            if acc_train > 0.6:
                class_temperture = target_pred_per_label.detach()
                records += 1
            for index in range(batch_size):
                source1_shared_weight[index, 0] = target_pred_per_label[label_source1[index], 0]
                source2_shared_weight[index, 0] = target_pred_per_label[label_source2[index], 0]
                target_shared_weight[index, 0] = target_margin[index, 0] * target_pred_per_label[target_pseudo_label[index], 0]
            source1_shared_weight = normalize_weight(source1_shared_weight, cut=args.train.cut, expand=True)
            source2_shared_weight = normalize_weight(source2_shared_weight, cut=args.train.cut, expand=True)
            target_shared_weight = normalize_weight(target_shared_weight, cut=args.train.cut, expand=True)

            source1_shared_label = torch.lt(label_source1, args.data.dataset.n_share1).view(batch_size, 1).float()
            source1_shared_index = torch.nonzero(source1_shared_label.flatten()).flatten()
            source1_private_label = torch.ge(label_source1, args.data.dataset.n_share_common).view(batch_size, 1).float()
            source1_private_index = torch.nonzero(source1_private_label.flatten()).flatten()
            source2_shared_label = torch.lt(label_source2, args.data.dataset.n_share_common).view(batch_size, 1).float()
            source2_shared_index = torch.nonzero(source2_shared_label.flatten()).flatten()
            source2_private_label = torch.ge(label_source2, args.data.dataset.n_share_common).view(batch_size, 1).float()
            source2_private_index = torch.nonzero(source2_private_label.flatten()).flatten()
            target_shared_label = torch.lt(label_target, args.data.dataset.n_share_common).view(batch_size, 1).float()
            target_shared_index = torch.nonzero(target_shared_label.flatten()).flatten()
            target_private_label = torch.ge(label_target, target_private_classes_num).view(batch_size, 1).float()
            target_private_index = torch.nonzero(target_private_label.flatten()).flatten()

        # ==============================compute loss
        cls_s1 = nn.CrossEntropyLoss()(fc2_s1, label_source1)
        cls_s2 = nn.CrossEntropyLoss()(fc2_s2, label_source2)
        cls_s = cls_s1 + cls_s2
        cls_s = cls_s

        domain_adv_loss = torch.mean(source1_shared_weight * nn.BCELoss(reduction='none')(domain_prob_source1, torch.ones_like(domain_prob_source1)), dim=0) + \
                          torch.mean(source2_shared_weight * nn.BCELoss(reduction='none')(domain_prob_source2, torch.ones_like(domain_prob_source2)), dim=0) + \
                          2 * torch.mean(target_shared_weight * nn.BCELoss(reduction='none')(domain_prob_target, torch.zeros_like(domain_prob_target)), dim=0)
        domain_adv_loss = domain_adv_loss / 2

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_domain_discriminator]):
            loss = cls_s + domain_adv_loss
            loss.backward()


        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            logger.add_scalar('loss/cls_s', cls_s, global_step)
            logger.add_scalar('loss/domain_adv_loss', domain_adv_loss, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:
            counters = [AccuracyCounter() for x in range(len(source_classes)+1)]
            with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
                 Accumulator(['feature_test', 'predict_prob_test', 'label', 'test_shared_weight']) as target_accumulator, \
                 torch.no_grad():
                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.cuda()
                    label = label.cuda()

                    feature_test = feature_extractor.forward(im)
                    feature_test, feature_short, before_softmax, predict_prob_test = classifier.forward(feature_test)
                    test_shared_weight = (predict_prob_test.max(1)[0] - torch.sort(predict_prob_test, dim=1, descending=True)[0][:, 1])

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            counters = [AccuracyCounter() for x in range(len(source_classes)+1)]

            for (each_predict_prob, each_test_shared_weight, each_label) in zip(predict_prob_test, test_shared_weight, label):
                each_pred_id = np.argmax(each_predict_prob)
                if each_label in source_classes:
                    counters[each_label].Ntotal += 1.0
                    if each_pred_id == each_label and each_test_shared_weight >= 0.5:
                        counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if each_test_shared_weight < 0.5:
                        counters[-1].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)
            acc_known = [x.reportAccuracy() for x in counters[:-1] if not np.isnan(x.reportAccuracy())]
            acc_known = torch.ones(1, 1) * np.mean(acc_known)
            acc_unknown = counters[-1].Ncorrect / (counters[-1].Ntotal + 1e-10)
            acc_unknown = torch.ones(1, 1) * acc_unknown

            logger.add_scalar('acc/acc_test', acc_test, global_step)
            logger.add_scalar('acc/acc_known', acc_known, global_step)
            logger.add_scalar('acc/acc_unknown', acc_unknown, global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'domain_discriminator': domain_discriminator.state_dict() if not isinstance(domain_discriminator, Nonsense) else 1.0,
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)

            sns_plot(source1_shared_weight, source2_shared_weight, target_shared_weight, source1_shared_index, source1_private_index, source2_shared_index, source2_private_index,
                     target_shared_index, target_private_index, global_step, name='step{}_w.png'.format(global_step), log=True)
