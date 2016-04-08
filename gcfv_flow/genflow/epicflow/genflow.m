IMAGE1_PATH='/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/1_1.jpg'
IMAGE2_PATH='/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/1_2.jpg'
FLOW_SAVE_PATH='/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/1_1.jpg'

f = get_epicflow(IMAGE1_PATH, IMAGE2_PATH, FLOW_SAVE_PATH);
