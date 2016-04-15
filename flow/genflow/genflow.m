function genflow(proc_rank, proc_size),


addpath(genpath('/lustre/yixi/face_segmentation_finetune/gcfv_flow/genflow/epicflow'))
addpath(genpath('utils'));

frame_dir_path = '/lustre/yueheing/datasets/camvid/raw_frames/';
frame_dirs = dir(frame_dir_path);
frame_dirs = {frame_dirs.name};

interested_file_dir = '/lustre/yixi/data/CamVid/701_StillsRaw_full/';
interested_files = dir(interested_file_dir);
interested_files = {interested_files(~[interested_files.isdir]).name};


SAVE_DIR = '/lustre/yixi/data/CamVid/flow_all';


for frame_dir = frame_dirs,
	if (strcmp(frame_dir{:},'.')==1) | (strcmp(frame_dir{:},'..')==1), continue; end

	fprintf('Working on video/folder %s\n', frame_dir{:});
	file_path = [frame_dir_path, frame_dir{:},'/%s.png']
	fprintf('file_path=%s\n', file_path);
	files = dir(sprintf(file_path, '*'));
	files = {files(~[files.isdir]).name};
	files = sort(files);


	% Should name as *.fk.flow_x.png or *.bk_flow_x.png
	deltas = [1,-1, 2, -2, 4, -4];
	for d_idx = 1:length(deltas),
		delta = deltas(d_idx);
		vv = code(delta);

		fprintf('Generating for delta=%d vv=%s\n', delta, vv);
		orimm=[];
		mm=[];
			
		FLOW_SAVE_DIR = [SAVE_DIR, '/flo/%s.', vv, '.flo'];
		FLOW_X_SAVE_DIR = [SAVE_DIR, '/flow/%s.', vv, '.flow_x.png']
		FLOW_Y_SAVE_DIR = [SAVE_DIR, '/flow/%s.', vv, '.flow_y.png']

		for i = 1:length(files),
			if mod(i, proc_size) == proc_rank,
				name = files{i};
				if ~ismember(name, interested_files), continue; end
				[~, name, ~] = fileparts(name);
				[pathstr, name1, ext1] = fileparts(sprintf(file_path, name));
				FLOW_SAVE_PATH = sprintf(FLOW_SAVE_DIR, name1);
				sprintf('flow_save_path=%s\n', FLOW_SAVE_PATH);
				
				if exist(FLOW_SAVE_PATH)~=2,
					if ((i+delta>=1) & (i+delta<=length(files))),
						name2 = files{i+delta};
						[~, name2, ~] = fileparts(name2);
						backward = 0;
					else,
						name2 = files{i-delta};
						[~, name2, ~] = fileparts(name2);
						backward = 1;
					end

					fprintf('name1=%s name2=%s\n', name1, name2);
					IMAGE1_PATH = sprintf(FRAME_DIR, name1);
					IMAGE2_PATH = sprintf(FRAME_DIR, name2);
					
					fprintf('Generating epicflow for %s %s  and save to %s\n', IMAGE1_PATH, IMAGE2_PATH, FLOW_SAVE_PATH);
					f = get_epicflow(IMAGE1_PATH, IMAGE2_PATH, FLOW_SAVE_PATH);
					
					if backward==1,
						flow_img = readFlowFile(f);
						writeFlowFile(flow_img*(-1), f);
					end
				end

				flow_img = readFlowFile(FLOW_SAVE_PATH);
				orimm = [orimm, mean(mean(flow_img(:,:,1)))];
				cut_k = 20;
				flow_img(flow_img<-cut_k) = -cut_k;
				flow_img(flow_img>cut_k) = cut_k;
				flow_img = uint8(round((flow_img+cut_k)/(2*cut_k)*255));
				imwrite(flow_img(:,:,1), sprintf(FLOW_X_SAVE_DIR, name1));
				imwrite(flow_img(:,:,2), sprintf(FLOW_Y_SAVE_DIR, name1));
				sprintf(FLOW_Y_SAVE_DIR, name1)	
				mm = [mm, mean(mean(flow_img(:,:,1)))];
				
			end
		end

		fprintf('mean val = where delta=%d %f\n', delta, mean(orimm));
		fprintf('mean val =%f\n', mean(mm));


	end % delta
end % FRAME_DIR

	function vv = code(delta),
		if delta>0, 
			vv='f'; 
		else, 
			vv='b'; 
		end
		vv=[vv, num2str(abs(delta))]
	end

end % function end
