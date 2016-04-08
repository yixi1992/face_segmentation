

addpath(genpath('epicflow'))

FRAME_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/%s.jpg';
FLOW_SAVE_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flo/%s.flo';
FLOW_X_SAVE_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/%s.flow_x.png';
FLOW_Y_SAVE_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/%s.flow_y.png';

files = dir(sprintf(FRAME_DIR, '*'));
files = files(~[files.isdir]);

for i=1:length(files)-1,
	[pathstr, name1, ext1] = fileparts(files(i).name);
	[pathstr, name2, ext2] = fileparts(files(i+1).name);
	if name1~=name2,
		continue;
	end

	IMAGE1_PATH = sprintf(FRAME_DIR, name1)
	IMAGE2_PATH = sprintf(FRAME_DIR, name2)
	FLOW_SAVE_PATH = sprintf(FLOW_SAVE_DIR, name1)

	f = get_epicflow(IMAGE1_PATH, IMAGE2_PATH, FLOW_SAVE_PATH);
	addpath(genpath('utils'));
	flow_img = readFlowFile(f);
	% width*height*2 real numbers double optical flow
	imwrite(flow_img(:,:,1), sprintf(FLOW_X_SAVE_DIR, name1));
	imwrite(flow_img(:,:,2), sprintf(FLOW_Y_SAVE_DIR, name1));
end



%{
FOLDER_PATH='/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/'
FLOW_SAVE_FOLDER_PATH='/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/'
dir_get_epicflow(FOLDER_PATH, FLOW_SAVE_FOLDER_PATH)
%}

