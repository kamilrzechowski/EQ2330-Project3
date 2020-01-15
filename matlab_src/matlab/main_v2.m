% Project 3 point 2
clear

%Uniform Quantizer 
step = [2^3 2^4 2^5 2^6];
FPS = 30;
n_frames = 50;
bSize = 16;
video_width = 176; video_height = 144;
% Import video luminance [video_width, vieo_height] 
Video = yuv_import_y('foreman_qcif.yuv',[video_width video_height],50);
frames = zeros(video_height,video_width,n_frames);               
for i=1:n_frames
    frames(:,:,i) = Video{i,1};
end

%init variables
err=zeros(length(Video),length(step));

RecoVideo = zeros(video_height,video_width,n_frames,length(step));
video_rate = zeros(n_frames,length(step));
%DCT-2
for f=1:length(Video)
    for q=1:length(step)
            ww = 1:bSize;
            hh = 1:bSize;
            block_num = 1; %block count for given quntization level and frame

            for h = 1:video_height/bSize
                for w = 1:video_width/bSize
                    [RecoVideo(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q), rate] = ...
                    codeBlock(frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f),step(q),bSize);
                video_rate(f,q) = video_rate(f,q) + rate;
                end
            end
            err(f,q) = distortion(RecoVideo(:,:,f,q),frames(:,:,f));
    end
end

%{
%test j=1 best qualiy, i=1 worse quality
Video_play = zeros(size(Video{1},1),size(Video{1},2),length(Video));
for i=1:length(Video)
    Video_play(:,:,i) = Video_rec{i,4};
end
implay(uint8(Video_play),FPS);
%}

meanPSNR = mean(PSNR(err));

ratesKBPS = mean(video_rate,1).*FPS/1000;

figure;
hold on;
plot(ratesKBPS, meanPSNR, 'o-');
grid on;
title('PSNR vs bitrate (DCT)');
grid on;
xlabel('Bit rate [kbps]');
ylabel('PSNR [dB]');
%xlim([0.2 1.6]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mother-daughter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Video = yuv_import_y('mother-daughter_qcif.yuv',[video_width video_height],50);
frames = zeros(video_height,video_width,n_frames);               
for i=1:n_frames
    frames(:,:,i) = Video{i,1};
end

%init variables
err=zeros(length(Video),length(step));

RecoVideo = zeros(video_height,video_width,n_frames,length(step));
video_rate = zeros(n_frames,length(step));
%DCT-2
for f=1:length(Video)
    for q=1:length(step)
            ww = 1:bSize;
            hh = 1:bSize;
            block_num = 1; %block count for given quntization level and frame

            for h = 1:video_height/bSize
                for w = 1:video_width/bSize
                    [RecoVideo(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q), rate] = ...
                    codeBlock(frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f),step(q),bSize);
                video_rate(f,q) = video_rate(f,q) + rate;
                end
            end
            err(f,q) = distortion(RecoVideo(:,:,f,q),frames(:,:,f));
    end
end

%{
%test j=1 best qualiy, i=1 worse quality
Video_play = zeros(size(Video{1},1),size(Video{1},2),length(Video));
for i=1:length(Video)
    Video_play(:,:,i) = Video_rec{i,4};
end
implay(uint8(Video_play),FPS);
%}

meanPSNR = mean(PSNR(err));

ratesKBPS = mean(video_rate,1).*FPS/1000;

plot(ratesKBPS, meanPSNR, 'o-');
grid on;
legend('foreman video','mother-daughter');
grid on;
xlabel('Bit rate [kbps]');
ylabel('PSNR [dB]');
hold off;
%xlim([0.2 1.6]);