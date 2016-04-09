function genflow(proc_rank, proc_size),


addpath(genpath('epicflow'))
addpath(genpath('utils'));

WORK_DIR = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos'
FRAME_DIR = fullfile(WORK_DIR, 'frames/%s.jpg');
FLOW_SAVE_DIR = fullfile(WORK_DIR, 'flo/%s.flo');
% Should name as *.fk.flow_x.png or *.bk_flow_x.png
FLOW_X_SAVE_DIR = fullfile(WORK_DIR, 'flow/%s.flow_x.png');
FLOW_Y_SAVE_DIR = fullfile(WORK_DIR, 'flow/%s.flow_y.png');

files = dir(sprintf(FRAME_DIR, '*'));
files = files(~[files.isdir]);


for i=1:length(files),
	if mod(i, proc_size) == proc_rank,
		[pathstr, name1, ext1] = fileparts(files(i).name);
		FLOW_SAVE_PATH = sprintf(FLOW_SAVE_DIR, name1);
		if exist(FLOW_SAVE_PATH)~=2,
			parts = strread(name1,'%s','delimiter','_');
			vid = str2num(parts{1});
			fid = str2num(parts{2});
			name2 = [num2str(vid), '_', num2str(fid+1)];
			
			backward = 0;
			if exist(sprintf(FRAME_DIR, name2), 'file')~=2,
				if exist(sprintf(FRAME_DIR, [num2str(vid), '_', num2str(fid-1)]))==2,
					name2 = [num2str(vid), '_', num2str(fid-1)];
					backward = 1;
				else
					print 'cant find', name1
					continue;
				end
			end

			IMAGE1_PATH = sprintf(FRAME_DIR, name1);
			IMAGE2_PATH = sprintf(FRAME_DIR, name2);

			f = get_epicflow(IMAGE1_PATH, IMAGE2_PATH, FLOW_SAVE_PATH);
			
			if backward==1,
				flow_img = readFlowFile(f);
				writeFlowFile(flow_img*(-1), f);
			end
		end

		flow_img = readFlowFile(FLOW_SAVE_PATH);
		flow_img(flow_img<-20) = 20;
		flow_img(flow_img>20) = 20;
		flow_img = uint8(round((flow_img+20)/40*255));
		imwrite(flow_img(:,:,1), sprintf(FLOW_X_SAVE_DIR, name1));
		imwrite(flow_img(:,:,2), sprintf(FLOW_Y_SAVE_DIR, name1));
		sprintf(FLOW_Y_SAVE_DIR, name1)		
	end
end

end

