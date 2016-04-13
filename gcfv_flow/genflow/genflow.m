function genflow(proc_rank, proc_size),


addpath(genpath('epicflow'))
addpath(genpath('utils'));

WORK_DIRS = {'/lustre/yixi/data/gcfv_dataset/cross_validation/videos', '/lustre/yixi/data/gcfv_dataset/external_validation/videos'}

for WORK_DIR = WORK_DIRS,
WORK_DIR


FRAME_DIR = [WORK_DIR{:}, '/frames/%s.jpg'];


% Should name as *.fk.flow_x.png or *.bk_flow_x.png
deltas = [-1, 2, -2, 4, -4];
for d_idx = 1:length(deltas),
delta = deltas(d_idx);
vv='';
if delta>0, 
	vv='f'; 
else, 
	vv='b'; 
end
vv=[vv, num2str(abs(delta))]

FLOW_SAVE_DIR = [WORK_DIR{:}, '/flo/%s.', vv, '.flo']
FLOW_X_SAVE_DIR = [WORK_DIR{:}, '/flow/%s.', vv, '.flow_x.png']
FLOW_Y_SAVE_DIR = [WORK_DIR{:}, '/flow/%s.', vv, '.flow_y.png']




files = dir(sprintf(FRAME_DIR, '*'));
files = files(~[files.isdir]);

orimm=[]
mm=[]


for i=1:length(files),
	if mod(i, proc_size) == proc_rank,
		[pathstr, name1, ext1] = fileparts(files(i).name);
		FLOW_SAVE_PATH = sprintf(FLOW_SAVE_DIR, name1);
		if exist(FLOW_SAVE_PATH)~=2,
			parts = strread(name1,'%s','delimiter','_');
			vid = str2num(parts{1});
			fid = str2num(parts{2});
			name2 = [num2str(vid), '_', num2str(fid+delta)];
			
			backward = 0;
			if exist(sprintf(FRAME_DIR, name2), 'file')~=2,
				if exist(sprintf(FRAME_DIR, [num2str(vid), '_', num2str(fid-delta)]))==2,
					name2 = [num2str(vid), '_', num2str(fid-delta)];
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
		orimm = [orimm, mean(mean(flow_img(:,:,1)))];
		flow_img(flow_img<-35) = 35;
		flow_img(flow_img>35) = 35;
		flow_img = uint8(round((flow_img+35)/(2*35)*255));
		imwrite(flow_img(:,:,1), sprintf(FLOW_X_SAVE_DIR, name1));
		imwrite(flow_img(:,:,2), sprintf(FLOW_Y_SAVE_DIR, name1));
		sprintf(FLOW_Y_SAVE_DIR, name1)	
		mm = [mm, mean(mean(flow_img(:,:,1)))];
		
	end
end

mean(orimm)
mean(mm)


end % delta
end % WORK_DIR
end % function end
