from utils.spc import *

args = config.get_config(model='unet_proj', max_size=128, dist='uniform')

def train(args, snapshot_path):
    """Training function for the semi-supervised learning model."""
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.batch_size % 2 == 0, "Batch size must be even for two-stream training"

    def create_model(ema=False):
        """Create the model for training."""
        model = net_factory(net_type=args.model, in_channels=1, num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    # Initialize the model and EMA model
    model = create_model()
    ema_model = create_model(ema=True)
    model.train()

    memory_wu = FeatureMemory(args.max_size, num_classes)
    memory_su = FeatureMemory(args.max_size, num_classes)

    # Initialize the dataloaders
    def worker_init_fn(worker_id):
        """Initialize the worker with a specific seed."""
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            WeakStrongAugment(args.patch_size)
        ])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    logging.info(f"Total slices is: {total_slices}, labeled slices is: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True
    )

    # Initialize the optimizer and loss functions
    optimizer = optim.SGD(
        model.parameters(), 
        lr=base_lr, 
        momentum=0.9, 
        weight_decay=0.0001
    )

    ce_loss = nn.CrossEntropyLoss()
    dc_loss = losses.MaskDiceLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, train_batch in enumerate(trainloader):
            image_w = train_batch['image_w'].float().cuda()
            _, image_wu = image_w.chunk(2)

            # Style-aware blending
            if args.dist == 'uniform':
                image_m = style_mixing_uniform(image_w.clone())
            else:
                raise ValueError('Unknown distribution type: {}'.format(args.dist))

            image_ml, _ = image_m.chunk(2)
            image_wm = torch.cat([image_ml, image_wu], dim=0)

            gt_w = train_batch['label_w'].long().cuda()
            gt_wl, _ = gt_w.chunk(2)

            image_s = train_batch['image_s'].float().cuda()
            _, image_su = image_s.chunk(2)

            # Labeled branch
            logit_w, feat_w = model(image_wm)
            logit_wl, _ = logit_w.chunk(2)
            _, feat_wu = feat_w.chunk(2)

            prob_w = logit_w.softmax(dim=1)
            prob_wl, prob_wu = prob_w.chunk(2)

            pred_w = torch.argmax(prob_w, dim=1)
            _, pred_wu = pred_w.chunk(2)

            loss_ce_l = ce_loss(logit_wl, gt_wl)
            loss_dc_l = dc_loss(prob_wl, gt_wl.unsqueeze(1))

            # Unlabeled branch
            logit_s, feat_s = model(image_s)
            _, feat_su = feat_s.chunk(2)

            prob_s = torch.softmax(logit_s, dim=1)
            _, prob_su = prob_s.chunk(2)

            loss_su = dc_loss(prob_su, pred_wu.unsqueeze(1))

            # Prototype-based cross-contrast learning
            with torch.no_grad():
                logit_wu_ema, _ = ema_model(image_wu)
                prob_wu_ema = torch.softmax(logit_wu_ema, dim=1)
                pred_wu_ema = prob_wu_ema.argmax(dim=1)

                logit_su_ema, _ = ema_model(image_su)
                prob_su_ema = torch.softmax(logit_su_ema, dim=1)
                pred_su_ema = prob_su_ema.argmax(dim=1)

            z1 = model.proj_head(feat_wu)  # [b, d, h, w]
            z2 = model.proj_head(feat_su)

            with torch.no_grad():
                memory_wu.enqueue(prob_wu.detach(), z1.detach())
                memory_su.enqueue(prob_su.detach(), z2.detach())
                proto_wu = memory_wu.calc_protos()  # [c, d]
                proto_su = memory_su.calc_protos()
                proto_norm_wu = nnF.normalize(proto_wu, dim=1)
                proto_norm_su = nnF.normalize(proto_su, dim=1)

            z1_norm = nnF.normalize(z1.permute(0, 2, 3, 1), dim=-1)
            z2_norm = nnF.normalize(z2.permute(0, 2, 3, 1), dim=-1)

            p1 = (z1_norm @ proto_norm_su.T).permute(0, 3, 1, 2)  # [b, c, h, w]
            loss_p1 = nnF.cross_entropy(p1, pred_wu_ema, reduction='mean')

            p2 = (z2_norm @ proto_norm_wu.T).permute(0, 3, 1, 2)
            loss_p2 = nnF.cross_entropy(p2, pred_su_ema, reduction='mean')

            loss_pc = (loss_p1 + loss_p2) / 2
            
            # Total loss
            loss_l = (loss_ce_l + loss_dc_l) / 2
            loss_u = 0.1 * loss_su + loss_pc
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = loss_l + consistency_weight * loss_u

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # Logging and saving metrics
            iter_num = iter_num + 1
            writer.add_scalar('info/loss', loss, iter_num)
            writer.add_scalar('info/loss_l', loss_l, iter_num)
            writer.add_scalar('info/loss_u', loss_u, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %.4f, loss_l: %.4f, loss_u: %.4f, loss_pc: %.4f' %
                (iter_num, loss, loss_l, loss_u, loss_pc))

            # Validation and model saving
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for val_batch in valloader:
                    metric_i = val_2d.test_single_volume(
                        val_batch["image"],
                        val_batch["label"],
                        model,
                        classes=num_classes,
                        patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(
                        snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                iterator.close()
                writer.close()
                return "Training Finished"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"log/{args.name}/{args.exp}_{args.labelnum}_{args.model}_{args.max_size}_{args.dist}_r{args.round}"
    os.makedirs(snapshot_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(snapshot_path, os.path.basename(__file__)))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
