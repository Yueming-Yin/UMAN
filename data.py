from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

'''
assume classes across domains are the same.
[0 1 ......................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
a, b, c, d, e, f, j = args.data.dataset.n_share1, args.data.dataset.n_source_private1, args.data.dataset.n_share2, args.data.dataset.n_source_private2, \
                      args.data.dataset.n_share_common, args.data.dataset.n_total, args.data.dataset.n_private_common
if args.data.dataset.source4 != None:
    g, h, l, m = args.data.dataset.n_share3, args.data.dataset.n_source_private3, args.data.dataset.n_share4, args.data.dataset.n_source_private4
    f = f - (e + b + d + h + m)
    e = int((a + c + g + l - e)/2)
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a + c + g + l - 2 * e for i in range(b)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source1 > 1:
        source_private_classes1 = [i + a + c + g + l - 2 * e + 5 for i in range(b)]
    common_classes2 = [i + a for i in range(c - 2 * e)]
    source_private_classes2 = [i + a + c + g + l - 2 * e + b for i in range(d)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source2 > 1:
        source_private_classes2 = [i + a + c + g + l - 2 * e + b + 5 for i in range(d)]
    common_classes3 = [i + a + c - e for i in range(g - e)]
    source_private_classes3 = [i + a + c + g + l - 2 * e + b + d for i in range(h-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source3 > 1:
        source_private_classes3 = [i + a + c + g + l - 2 * e + b + d + 5 for i in range(h - j)]
    common_classes4 = [i + a + c + g - e for i in range(l - e)]
    source_private_classes4 = [i + a + c + g + l - 2 * e + b + d + h for i in range(m - j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source4 > 1:
        source_private_classes4 = [i + a + c + g + l - 2 * e + b + d + h + 5 for i in range(m - j)]
    target_private_classes = [i + a + c + g + l - 2 * e + b + d + h + m for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target > 1:
        target_private_classes = [i + a + c + g + l - 2 * e + b + d + h + m + 5 for i in range(f)]
    source_classes1 = common_classes1 + source_private_classes1
    source_classes2 = common_classes2 + source_private_classes2
    source_classes3 = common_classes3 + source_private_classes3
    source_classes4 = common_classes4 + source_private_classes4
    source_classes = common_classes1 + common_classes2 + common_classes3 + common_classes4 + source_private_classes1 + source_private_classes2 + source_private_classes3 + source_private_classes4
    target_classes = common_classes1 + common_classes2 + common_classes3 + common_classes4 + target_private_classes
    print('source_classes1 =', source_classes1, '\n', 'source_classes2 =', source_classes2, '\n', 'source_classes3 =', source_classes3, '\n', 'source_classes4 =', source_classes4
          , '\n', 'source_classes =', source_classes, '\n', 'target_classes =', target_classes)
elif args.data.dataset.source3 != None:
    g, h = args.data.dataset.n_share3, args.data.dataset.n_source_private3
    f = f - (e + b + d + h)
    e = int((a + c + g - e)/2)
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a + c + g - 2 * e for i in range(b)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source1 > 1:
        source_private_classes1 = [i + a + c + g - 2 * e + 5 for i in range(b)]
    common_classes2 = [i + a for i in range(c - 2 * e)]
    source_private_classes2 = [i + a + c + g - 2 * e + b for i in range(d)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source2 > 1:
        source_private_classes2 = [i + a + c + g - 2 * e + b + 5 for i in range(d)]
    common_classes3 = [i + a + c - e for i in range(g - e)]
    source_private_classes3 = [i + a + c + g - 2 * e + b + d for i in range(h-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source3 > 1:
        source_private_classes3 = [i + a + c + g - 2 * e + b + d + 5 for i in range(h - j)]
    shared_private_classes23 = [i + a + c + g - 2 * e + b + d - j for i in range(j)]
    common_shared_classes12 = [i + a - e for i in range(e)]
    common_shared_classes23 = [i + a + c - 2 * e for i in range(e)]
    target_private_classes = [i + a + c + g - 2 * e + b + d + h for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target > 1:
        target_private_classes = [i + a + c + g - 2 * e + b + d + h + 5 for i in range(f)]
    source_classes1 = common_classes1 + common_shared_classes12 + source_private_classes1
    source_classes2 = common_shared_classes12 + common_classes2 + common_shared_classes23 + source_private_classes2
    source_classes3 = common_shared_classes23 + common_classes3 + shared_private_classes23 + source_private_classes3
    source_classes = common_classes1 + common_shared_classes12 + common_classes2 + common_shared_classes23 + common_classes3 + source_private_classes1 + source_private_classes2 + shared_private_classes23 + source_private_classes3
    target_classes = common_classes1 + common_shared_classes12 + common_classes2 + common_shared_classes23 + common_classes3 + target_private_classes
    print('source_classes1 =', source_classes1, '\n', 'source_classes2 =', source_classes2, '\n', 'source_classes3 =', source_classes3, '\n', 'source_classes =', source_classes, '\n', 'target_classes =', target_classes)
elif args.data.dataset.source2 != None:
    f = f - (e + b + d - j)
    e = a + c - e
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a + c - e for i in range(b-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source1 > 1:
        source_private_classes1 = [i + a + c - e + 5 for i in range(b-j)]
    common_classes2 = [i + a for i in range(c - e)]
    source_private_classes2 = [i + a + c - e + b for i in range(d-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source2 > 1:
        source_private_classes2 = [i + a + c - e + b + 5 for i in range(d-j)]
    common_shared_classes = [i + a - e for i in range(e)]
    private_shared_classes = [i + a + c - e + b - j for i in range(j)]
    target_private_classes = [i + a + c - e + b + d - j for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target > 1:
        target_private_classes = [i + a + c - e + b + d - j + 5 for i in range(f)]
    source_classes1 = common_classes1 + common_shared_classes + source_private_classes1 + private_shared_classes
    source_classes2 = common_shared_classes + common_classes2 + private_shared_classes + source_private_classes2
    source_classes = common_classes1 + common_shared_classes + common_classes2 + source_private_classes1 + private_shared_classes + source_private_classes2
    target_classes = common_classes1 + common_shared_classes + common_classes2 + target_private_classes
    print('source_classes1 =', source_classes1, '\n', 'source_classes2 =', source_classes2, '\n', 'source_classes =', source_classes, '\n', 'target_classes =', target_classes)
else:
    f = f - (e + b - j)
    e = a - e
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a - e for i in range(b-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.source1 > 1:
        source_private_classes1 = [i + a - e + 5 for i in range(b-j)]
    target_private_classes = [i + a - e + b - j for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target > 1:
        target_private_classes = [i + a - e + b - j + 5 for i in range(f)]
    source_classes1 = common_classes1 + source_private_classes1
    source_classes = source_classes1
    target_classes = common_classes1 + target_private_classes
    print('source_classes =', source_classes, '\n', 'target_classes =', target_classes)

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

source1_train_ds = FileListDataset(list_path=source1_file, path_prefix=dataset.prefixes[args.data.dataset.source1],
                            transform=train_transform, filter=(lambda x: x in source_classes1))
source1_test_ds = FileListDataset(list_path=source1_file,path_prefix=dataset.prefixes[args.data.dataset.source1],
                            transform=test_transform, filter=(lambda x: x in source_classes1))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))

classes1 = source1_train_ds.labels
freq1 = Counter(classes1)
class_weight1 = {x : 1.0 / freq1[x] if args.data.dataloader.class_balance else 1.0 for x in freq1}
source1_weights = [class_weight1[x] for x in source1_train_ds.labels]
sampler1 = WeightedRandomSampler(source1_weights, len(source1_train_ds.labels))

source1_train_dl = DataLoader(dataset=source1_train_ds, batch_size=int(args.data.dataloader.batch_size),
                             sampler=sampler1, num_workers=args.data.dataloader.data_workers, drop_last=True)
source1_test_dl = DataLoader(dataset=source1_test_ds, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)

if args.data.dataset.source2 != None:
    source2_train_ds = FileListDataset(list_path=source2_file, path_prefix=dataset.prefixes[args.data.dataset.source2],
                                       transform=train_transform, filter=(lambda x: x in source_classes2))
    source2_test_ds = FileListDataset(list_path=source2_file, path_prefix=dataset.prefixes[args.data.dataset.source2],
                                      transform=test_transform, filter=(lambda x: x in source_classes2))
    classes2 = source2_train_ds.labels
    freq2 = Counter(classes2)
    class_weight2 = {x: 1.0 / freq2[x] if args.data.dataloader.class_balance else 1.0 for x in freq2}
    source2_weights = [class_weight2[x] for x in source2_train_ds.labels]
    sampler2 = WeightedRandomSampler(source2_weights, len(source2_train_ds.labels))

    source2_train_dl = DataLoader(dataset=source2_train_ds, batch_size=int(args.data.dataloader.batch_size),
                                  sampler=sampler2, num_workers=args.data.dataloader.data_workers, drop_last=True)
    source2_test_dl = DataLoader(dataset=source2_test_ds, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                                 num_workers=1, drop_last=False)

if args.data.dataset.source3 != None:
    source3_train_ds = FileListDataset(list_path=source3_file, path_prefix=dataset.prefixes[args.data.dataset.source3],
                                       transform=train_transform, filter=(lambda x: x in source_classes3))
    source3_test_ds = FileListDataset(list_path=source3_file, path_prefix=dataset.prefixes[args.data.dataset.source3],
                                      transform=test_transform, filter=(lambda x: x in source_classes3))

    classes3 = source3_train_ds.labels
    freq3 = Counter(classes3)
    class_weight3 = {x: 1.0 / freq3[x] if args.data.dataloader.class_balance else 1.0 for x in freq3}
    source3_weights = [class_weight3[x] for x in source3_train_ds.labels]
    sampler3 = WeightedRandomSampler(source3_weights, len(source3_train_ds.labels))

    source3_train_dl = DataLoader(dataset=source3_train_ds, batch_size=int(args.data.dataloader.batch_size),
                                  sampler=sampler3, num_workers=args.data.dataloader.data_workers, drop_last=True)
    source3_test_dl = DataLoader(dataset=source3_test_ds, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                                 num_workers=1, drop_last=False)

if args.data.dataset.source4 != None:
    source4_train_ds = FileListDataset(list_path=source4_file, path_prefix=dataset.prefixes[args.data.dataset.source4],
                                       transform=train_transform, filter=(lambda x: x in source_classes4))
    source4_test_ds = FileListDataset(list_path=source4_file, path_prefix=dataset.prefixes[args.data.dataset.source4],
                                      transform=test_transform, filter=(lambda x: x in source_classes4))

    classes4 = source4_train_ds.labels
    freq4 = Counter(classes4)
    class_weight4 = {x: 1.0 / freq4[x] if args.data.dataloader.class_balance else 1.0 for x in freq4}
    source4_weights = [class_weight4[x] for x in source4_train_ds.labels]
    sampler4 = WeightedRandomSampler(source4_weights, len(source4_train_ds.labels))

    source4_train_dl = DataLoader(dataset=source4_train_ds, batch_size=int(args.data.dataloader.batch_size),
                                  sampler=sampler4, num_workers=args.data.dataloader.data_workers, drop_last=True)
    source4_test_dl = DataLoader(dataset=source4_test_ds, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                                 num_workers=1, drop_last=False)