config = dict(
    model=dict(
        type='FreeNet',
        params=dict(
            in_channels=249,
            num_classes=7,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewPaviaULoader',
            params=dict(
                num_workers=0,
                image_mat_path="/data/ljt/shenyang_data/VNIR_sample/43_obj.dat",
                gt_mat_path="/data/ljt/shenyang_data/label/43_obj1-train.tif",
                training=True,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='NewPaviaULoader',
            params=dict(
                num_workers=0,
                image_mat_path="/data/ljt/shenyang_data/VNIR_sample/43_obj.dat",
                gt_mat_path="/data/ljt/shenyang_data/label/43_obj1-test.tif",
                training=False,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.001
        )
    ),
    learning_rate=dict(
        type='ExponentialLR',
        params=dict(
            base_lr=0.001,
            power=0.9,
            max_iters=1000),
    ),
    out_config=dict(
        params=dict(
            image_size=(940, 475),
            palette=[
                [0, 0, 0],
                [255, 0, 0],
                [255, 255, 255],
                [176, 48, 96],
                [255, 255, 0],
                [255, 127, 80],
                [0, 255, 0],
#                [0, 205, 0],
#                [0, 139, 0],
#                [127, 255, 212],
#                [160, 32, 240],
#                [216, 191, 216],
#                [0, 0, 255],
#                [0, 0, 139],
#                [218, 112, 214],
#                [160, 82, 45],
#                [0, 255, 255],
#                [255, 165, 0],
#                [127, 255, 0],
#                [139, 139, 0],
#                [0, 139, 139],
#                [205, 181, 205],
#                [238, 154, 0]
            ],
            save_path="./log"
        ),
    )
)
