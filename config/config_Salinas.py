config = dict(
    model=dict(
        type='FreeNet',
        params=dict(
            in_channels=204,
            num_classes=16,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewSalinasLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./img/Salinas_corrected.mat',
                gt_mat_path='./img/Salinas_gt.mat',
                training=True,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='NewSalinasLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./img/Salinas_corrected.mat',
                gt_mat_path='./img/Salinas_gt.mat',
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
            max_iters=100),
    ),
    out_config=dict(
        params=dict(
            image_size=(512, 217),
            palette=[
                [0, 0, 0],
                [255, 250, 205],
                [0, 0, 255],
                [255, 0, 0],
                [0, 255, 127],
                [255, 0, 255],
                [25, 25, 112],
                [100, 149, 237],
                [139, 134, 78],
                [50, 205, 50],
                [153, 50, 204],
                [0, 134, 139],
                [72, 61, 139],
                [124, 205, 124],
                [139, 69, 19],
                [32, 178, 170],
                [255, 255, 0]],
            save_path="./log"
        ),
    )
)
