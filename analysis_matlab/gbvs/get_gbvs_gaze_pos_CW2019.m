clear all
close all
% gbvs_install

%   C is for Color
%   I is for Intensity
%   O is for Orientation
%   R is for contRast
%   F is for Flicker
%   M is for Motion
%   D is for DKL Color (Derrington Krauskopf Lennie) == much better than C channel
channels_list = {'IORFMD', 'I', 'O', 'R', 'F', 'M', 'D'};
channels_label = {'All', 'Intensity', 'Orientation', 'Contrast', 'Flicker', 'Motion', 'DKL_Color'};
num_channels = length(channels_list);

video_info = dir("clips_png");
video_names = {video_info(3:end).name};
num_videos = length(video_names);
%% 

for video_idx = 1:num_videos
    video_name = char(video_names(video_idx));
    disp(video_name)

    video_dir = strcat("clips_png/", video_name);
    num_frames = length(dir(video_dir))-2;
    
    % get video sequence name
    %num_frames = 3; % change
    fname = cell(num_frames, 1);
    for frame_idx = 1:num_frames
        fname{frame_idx} = strcat(video_dir, sprintf('/%04d.png', frame_idx));
    end

    gbvs_res = struct;
    for i = 1:num_channels
        channel = char(channels_list(i));
        channel_label = char(channels_label(i));
        disp(channel_label)
    
        % compute the saliency maps for this sequence
        param = makeGBVSParams; % get default GBVS params
        param.channels = channel; 
        
        motinfo = [];           % previous frame information, initialized to empty
        %out = cell(num_frames, 1);
        gaze_pos = zeros(num_frames,2);
        for frame_idx=1:num_frames %num_frames
            file_name = fname{frame_idx};
            disp(file_name);
            img = imread(file_name);
            [out, motinfo] = gbvs(img, param, motinfo);
            %[out{frame_idx}, motinfo] = gbvs(img, param, motinfo);
            map = out.master_map_resized;
            [M,I] = max(map,[],"all","linear");
            [ey, ex] = ind2sub(size(map),I);
            gaze_pos(frame_idx, :) = [ex, ey];
        end
        gbvs_res.(channel_label) = gaze_pos;
    end
    
    save(strcat('analysis/gbvs_gaze_CW2019_results/', video_name, '.mat'), 'gbvs_res');
end