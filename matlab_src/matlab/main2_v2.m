%projct 3 point 3
clear

%Uniform Quantizer 
step = [2^3 2^4 2^5 2^6];
FPS = 30;
video_width = 176; video_height = 144;
n_frames = 50;

% Load video [video_width, vieo_height] 
Video = yuv_import_y('foreman_qcif.yuv',[video_width video_height],n_frames);
%Video = yuv_import_y('mother-daughter_qcif.yuv',[video_width video_height],n_frames);
frames = zeros(video_height,video_width,n_frames);               
for i=1:n_frames
    frames(:,:,i) = Video{i,1};
end

% Block Size for replacement
bSize = 16;
n_blocks = video_width*video_height/(bSize^2);

% Lagrange multiplier for optimization in choice of coding mode
% To be tuned so that DistortionCOst roughly == RateCost
lambda1 = 0.001*(step.^2);
lambda2 = 0.002*(step.^2);

video_rate1 = zeros(n_frames,length(step));
RecoVideo1 = zeros(video_height,video_width,n_frames,length(step));
PSNR1 = zeros(n_frames,length(step));
%code the first frame, becouse we cannot do better in this case
RecoVideo2 = zeros(video_height,video_width,n_frames,length(step));
video_rate2 = zeros(n_frames,length(step));
PSNR2 = zeros(n_frames,length(step));
for q=1:length(step)
        ww = 1:bSize;
        hh = 1:bSize;
        block_num = 1; %block count for given quntization level and frame

        for h = 1:video_height/bSize
            for w = 1:video_width/bSize
                [RecoVideo2(bSize*(h-1)+hh,bSize*(w-1)+ww,1,q), rate] = ...
                    codeBlock(frames(bSize*(h-1)+hh,bSize*(w-1)+ww,1),step(q),bSize);
                video_rate2(1,q) = video_rate2(1,q) + rate;
            end
        end
        RecoVideo1(1,q) = RecoVideo2(1,q);
        video_rate1(1,q) = video_rate2(1,q);
        PSNR2(1,q) = PSNR(distortion(RecoVideo2(:,:,1,q),frames(:,:,1)));
        PSNR1(1,q) = PSNR2(1,q);
end

num_replBocks = zeros(1,n_frames,length(step));
num_replBocks(:,1,:) = 0;  % cannot copy the first frame
%for each frame
for f=2:n_frames
    %for each quantization step
    for q=1:length(step)
        ww = 1:bSize;
        hh = 1:bSize;
        block_num = 1; %block count for given quntization level and frame

        for h = 1:video_height/bSize
            for w = 1:video_width/bSize
                [bCoded, R1] = codeBlock(frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f),step(q),bSize);
                
                Dist1 = distortion(bCoded, frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f));
                Dist2 = distortion(RecoVideo2(bSize*(h-1)+hh,bSize*(w-1)+ww,f-1,q),...
                    frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f));
                
                R2 = 1; %copy one bit      
                R1 = R1 + 1;    %one aditional bit for mode selection
                % Choose mode that minimizes Lagrangian cost
                Cost1 = Dist1 + lambda1(q)*(R1); 
                Cost2 = Dist2 + lambda2(q)*R2;
                Costf = [Cost1 Cost2];
                [MinCost,ChosenMode] = min(Costf);
                
                 % Encode video (decide intramode or copy mode)
                if ChosenMode == 1    %intra mode - set to Reco1
                    RecoVideo2(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = bCoded;
                    video_rate2(f,q) = video_rate2(f,q) + R1;
                elseif ChosenMode == 2  %copy block from previous frame
                    RecoVideo2(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = ...
                        RecoVideo2(bSize*(h-1)+hh,bSize*(w-1)+ww,f-1,q);
                    % Save mode selected for visualizations and insight
                    num_replBocks(1,f,q) = num_replBocks(1,f,q) + 1;
                    video_rate2(f,q) = video_rate2(f,q) + R2;
                end  
                block_num = block_num + 1;
                
                %for mode 1
                RecoVideo1(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = bCoded;
                video_rate1(f,q) = video_rate1(f,q) + R1 - 1;
            end
        end
        PSNR2(f,q) = PSNR(distortion(RecoVideo2(:,:,f,q),frames(:,:,f)));
        PSNR1(f,q) = PSNR(distortion(RecoVideo1(:,:,f,q),frames(:,:,f)));
    end
end

Rate1 = mean(video_rate1,1);
Rate1 = (Rate1*FPS)/1000;
PSNR1_video = mean(PSNR1,1);
Rate2 = mean(video_rate2,1);
Rate2 = (Rate2*FPS)/1000;
PSNR2_video = mean(PSNR2,1);
plot(Rate1,PSNR1_video,'o-');
hold on;
plot(Rate2,PSNR2_video,'o-');
hold off;
grid on;
title('PSNR vs bit rate for conditional ceplenishment video coder (foreman)');
legend('intra mode','copy mode');
xlabel('Bit rate [kbps]');
ylabel('PSNR');

replacedBlock_count = zeros(1,length(step));
totalBock_count = zeros(1,length(step));
encodedBlock_count = zeros(1,length(step));
for q = 1:length(step)
    for f = 1:n_frames
        replacedBlock_count(q) = replacedBlock_count(q) + num_replBocks(1,f,q);
    end
    totalBock_count(q) = n_blocks*n_frames;
    encodedBlock_count(q) = totalBock_count(q) - replacedBlock_count(q);
end
plot_mat = [replacedBlock_count(:),encodedBlock_count(:)];
figure;
bar(step,plot_mat);
legend('number of copied 16x16 blocks','number of 16x16 intra frame coded blocks');
title('Number of copied blockes vs number of coded blocks for foreman video');
xlabel('Quzantization step');
ylabel('Number of 16x16 Blocks');

%%
% visualise
%implay(uint8(RecoVideo2(:,:,:,3)),FPS);