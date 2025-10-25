
# Collect activation scales.
python sq_collect.py configs/release/x_0.py --load-from ../artifacts/release/x_0/checkpoint_epoch_1
python sq_collect.py configs/release/x_1.py --load-from ../artifacts/release/x_1/checkpoint_epoch_1
python sq_collect.py configs/release/y_0.py --load-from ../artifacts/release/y_0/checkpoint_epoch_1
python sq_collect.py configs/release/y_1.py --load-from ../artifacts/release/y_1/checkpoint_epoch_1
python sq_collect.py configs/release/x_2infer.py --load-from ../artifacts/release/x_2/checkpoint_epoch_1
python sq_collect.py configs/release/y_2infer.py --load-from ../artifacts/release/y_2/checkpoint_epoch_1

# Convert weights to W8A8.
python sq_convert.py configs/release/x_0.py
python sq_convert.py configs/release/x_1.py
python sq_convert.py configs/release/y_0.py
python sq_convert.py configs/release/y_1.py
python sq_convert.py configs/release/x_2infer.py
python sq_convert.py configs/release/y_2infer.py

# Check score of W8A8 models
python train.py configs/release/w8a8/x_0.py --load-from ../artifacts/release/x_0/checkpoint_w8a8/ --eval-only
# ...

# Prepare submission.
python prepare_submission.py configs/release/w8a8/x_0.py --load-from ../artifacts/release/x_0/checkpoint_w8a8/
python prepare_submission.py configs/release/w8a8/x_1.py --load-from ../artifacts/release/x_1/checkpoint_w8a8/
python prepare_submission.py configs/release/w8a8/y_0.py --load-from ../artifacts/release/y_0/checkpoint_w8a8/
python prepare_submission.py configs/release/w8a8/y_1.py --load-from ../artifacts/release/y_1/checkpoint_w8a8/
python prepare_submission.py configs/release/w8a8/x_2infer.py --load-from ../artifacts/release/x_2/checkpoint_w8a8/
python prepare_submission.py configs/release/w8a8/y_2infer.py --load-from ../artifacts/release/y_2/checkpoint_w8a8/