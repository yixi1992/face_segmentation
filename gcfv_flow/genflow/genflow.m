function genflow(proc_rank, proc_size),

addpath(genpath('epicflow'))
addpath(genpath('utils'));

FRAME_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/%s.jpg';
FLOW_SAVE_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flo/%s.flo';
FLOW_X_SAVE_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/%s.flow_x.png';
FLOW_Y_SAVE_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/%s.flow_y.png';

files = dir(sprintf(FRAME_DIR, '*'));
files = files(~[files.isdir]);


for i=1:length(files),
	if mod(i, proc_size) == proc_rank,
		[pathstr, name1, ext1] = fileparts(files(i).name);
		parts = strread(name1,'%s','delimiter','_');
		vid = str2num(parts{1});
		fid = str2num(parts{2});
		name2 = [num2str(vid), '_', num2str(fid+1)];
		
		backward = 0;
		if exist(sprintf(FRAME_DIR, name2), 'file')~=2,
			if exist(sprintf(FRAME_DIR, [num2str(vid), '_', num2str(fid-1)]))==2,
				name2 = sprintf(FRAME_DIR, [num2str(vid), '_', num2str(fid-1)]);
				backward = 1;
			else
				continue;
			end
		end

		IMAGE1_PATH = sprintf(FRAME_DIR, name1)
		IMAGE2_PATH = sprintf(FRAME_DIR, name2)
		FLOW_SAVE_PATH = sprintf(FLOW_SAVE_DIR, name1)

		f = get_epicflow(IMAGE1_PATH, IMAGE2_PATH, FLOW_SAVE_PATH);
		
		if backward==1,
			flow_img = readFlowFile(f);
			writeFlowFile(flow_img*(-1), f);

			%imwrite(flow_img(:,:,1), sprintf(FLOW_X_SAVE_DIR, name1));
			%imwrite(flow_img(:,:,2), sprintf(FLOW_Y_SAVE_DIR, name1));
		end

	end
end

end

