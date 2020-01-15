%projct 3 point 3
clear

%Uniform Quantizer 
step = [2^3 2^4 2^5 2^6];
FPS = 30;
video_width = 176; video_height = 144;
n_frames = 50;

% Load video [video_width, vieo_height] 
Video = yuv_import_y('foreman_qcif.yuv',[video_width video_height],n_frames);
frames = zeros(video_height,video_width,n_frames);               
for i=1:n_frames
    frames(:,:,i) = Video{i,1};
end

% Block Size for replacement
bSize = 16;
n_blocks = video_width*video_height/(bSize^2);

% Lagrange multiplier for optimization in choice of coding mode
% To be tuned so that DistortionCOst roughly == RateCost
lambda = 0.02*(step.^2);


%%%
%code for conditional replacement
%There are two modes. Mode 1 use intrframe coding. Mode 2 use block coping
%if Lagrangian cost function is smaller than for mode 1
%%%
framesDCTq = zeros(video_height,video_width,n_frames,length(step));
%variables for mode 1
Reco1 = zeros(video_height,video_width,n_frames,length(step));
MSE1 = zeros(n_frames,length(step));
PSNR1 = zeros(n_frames,length(step));
Rate1 = zeros(n_frames,length(step));
%variables for mode 2
Reco2 = zeros(video_height,video_width,n_frames,length(step));
MSE2 = zeros(n_frames,length(step));
PSNR2 = zeros(n_frames,length(step));
Rate2 = zeros(n_frames,length(step));
%intraframe bitratecalculation
for f=1:n_frames
    %Intra frame only (mode 1)
    DCT_frame = comp_dct(frames(:,:,f),8);
    for q=1:length(step)
        framesDCTq(:,:,f,q) = quantizer(DCT_frame,step(q));
        Reco1(:,:,f,q) = comp_idct(framesDCTq(:,:,f,q),8);
        %Compute MSEs and PSNRs for mode 1
        MSE1(f,q) = distortion(Reco1(:,:,f,q),frames(:,:,f));
        PSNR1(f,q) = PSNR(MSE1(f,q));
        %Compute bits/coeff for mode 1
        Rate1(f,q) = BitRate(framesDCTq(:,:,f,q), step(q));
    end
end

% diffrent modes have different bitrate (bits per block)
% mode 1 intrafram coding
% mode 2 block coping
R1 = bit_rate_estimation(framesDCTq,n_frames,step,video_width, video_height)*bSize^2 + 1;   %mean(Rate1,1)*bSize^2 + 1;        %intra mode bitrate
R2 = ones(1,length(step));            %copy mode bitrate
Rates = [R1(:),R2(:)];
nBits_replMode = zeros(n_frames,length(step)); %bits for encoding each frame 
nBits_replMode(1,:) = R1*n_blocks;  %bits used for intra mode of 1st frame

% variabe for performance visualisation (how many blocks have been
% repleaced
%(count_of_copied_blocks, frame_num, quantization_level)
num_replBocks = zeros(1,n_frames,length(step));
num_replBocks(:,1,:) = 0;  % cannot copy the first frame


% First frame can't copy bloks from pevoious frmae, so let's set intraframe
% coding for frist rframe for all quantzation levels
for q=1:length(step)
    Reco2(:,:,1,q) = Reco1(:,:,1,q);
end

%for each frame
for f=1:n_frames-1
    %for each quantization step
    for q=1:length(step)
        ww = 1:bSize;
        hh = 1:bSize;
        block_num = 1; %block count for given quntization level and frame
        
        %
        for h = 1:video_height/bSize
            for w = 1:video_width/bSize
                
                % For modes 1 and 2 distortions for encoding the next block
                Dist1 = distortion(frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f+1),...
                    Reco1(bSize*(h-1)+hh,bSize*(w-1)+ww,f+1,q));
                Dist2 = distortion(frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f+1),...
                    Reco2(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q));
                
                % Choose mode that minimizes Lagrangian cost
                Cost1 = Dist1 + lambda(q)*R1(q); 
                Cost2 = Dist2 + lambda(q)*R2(q);
                Costf = [Cost1 Cost2];
                [MinCost,ChosenMode] = min(Costf);
                
                % Accumulate Nbits used to get total Bits/Frame
                nBits_replMode(f+1,q) = nBits_replMode(f+1,q) + Rates(q,ChosenMode);
                
                % Encode video (decide intramode or copy mode)
                if ChosenMode == 1    %intra mode - set to Reco1
                    Reco2(bSize*(h-1)+hh,bSize*(w-1)+ww,f+1,q) = ...
                        Reco1(bSize*(h-1)+hh,bSize*(w-1)+ww,f+1,q);
                elseif ChosenMode == 2  %copy block from previous frame
                    Reco2(bSize*(h-1)+hh,bSize*(w-1)+ww,f+1,q) = ...
                        Reco2(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q);
                    % Save mode selected for visualizations and insight
                    num_replBocks(1,f+1,q) = num_replBocks(1,f+1,q) + 1;
                end  
                block_num = block_num + 1;          
            end
        end
        PSNR2(f,q) = PSNR(distortion(Reco2(:,:,f,q),frames(:,:,f)));
        
    end
end

% plot bit rate vs PSNR
%bit_rate_per_frame = zeros(length(step),1);
PSNR_video = mean(PSNR2,1);
%for q = 1:length(step)
%    for f = 1:n_frames
%        bit_rate_per_frame(q) = bit_rate_per_frame(q) + nBits_replMode(f,q);
%    end
%end
tmp = mean(nBits_replMode(:,:),1);
bit_rate_per_frame = mean(nBits_replMode(:,:),1);
figure;
%bit rate in kbps
bit_rate_per_frame = (bit_rate_per_frame*FPS)/1000;
plot(bit_rate_per_frame,PSNR_video, 'linewidth', 2);
title('PSNR vs bit rate for conditional ceplenishment video coder');
xlabel('Bit rate [kbps]');
ylabel('PSNR');


replacedBlock_count = zeros(1,length(step));
totalBock_count = zeros(1,length(step));
for q = 1:length(step)
    for f = 1:n_frames
        replacedBlock_count(q) = replacedBlock_count(q) + num_replBocks(1,f,q);
    end
    totalBock_count(q) = n_blocks*n_frames;
end
plot_mat = [replacedBlock_count(:),totalBock_count(:)];
figure;
bar(step,plot_mat);
legend('number of copied 16x16 blocks','total number of 16x16 blocks in the video');
title('Number of copied blockes vs total number of blocks for different quantization steps');
xlabel('Quzantization step');
ylabel('Number of 16x16 Blocks');

%%
% visualise
implay(uint8(Reco2(:,:,:,1)),FPS);