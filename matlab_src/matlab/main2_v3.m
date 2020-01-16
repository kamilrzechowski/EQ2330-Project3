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

%%%%%%%%%%%%%
% Motion mode preprocesing
%%%%%%%%%%%%%
max_shift = 10;
num_shift_comb = (2*max_shift + 1)^2;
shift_comb = zeros(2,num_shift_comb);
indx = 1;
for dx = -max_shift:1:max_shift
    for dy = -max_shift:1:max_shift
        shift_comb(1,indx) = dx;
        shift_comb(2,indx) = dy;
        indx = indx + 1;
    end
end

% Block Size for replacement
bSize = 16;
n_blocks = video_width*video_height/(bSize^2);

% Lagrange multiplier for optimization in choice of coding mode
% To be tuned so that DistortionCOst roughly == RateCost
%lambda1 = 0.0015*(step.^2);
%lambda2 = 0.2*(step.^2);
lambda1 = 0.001*(step.^2);
lambda2 = 0.002*(step.^2);

lambda11 = 0.001*(step.^2);
lambda22 = 0.002*(step.^2);
lambda33 = 0.0007*(step.^2);

%%%%%%%%%%%%%
% First frame
%%%%%%%%%%%%%
%onlu intra mode
video_rate1 = zeros(n_frames,length(step));
RecoVideo1 = zeros(video_height,video_width,n_frames,length(step));
PSNR1 = zeros(n_frames,length(step));
%only intra and copy modes
RecoVideo2 = zeros(video_height,video_width,n_frames,length(step));
video_rate2 = zeros(n_frames,length(step));
PSNR2 = zeros(n_frames,length(step));
%intra, copy and moution compensation mode
RecoVideo3 = zeros(video_height,video_width,n_frames,length(step));
video_rate3 = zeros(n_frames,length(step));
PSNR3 = zeros(n_frames,length(step));
%code the first frame, becouse we cannot do better in this case
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
        RecoVideo1(:,:,1,q) = RecoVideo2(:,:,1,q);
        video_rate1(1,q) = video_rate2(1,q);
        RecoVideo3(:,:,1,q) = RecoVideo2(:,:,1,q);
        video_rate3(1,q) = video_rate2(1,q);
        PSNR2(1,q) = PSNR(distortion(RecoVideo2(:,:,1,q),frames(:,:,1)));
        PSNR1(1,q) = PSNR2(1,q);
        PSNR3(1,q) = PSNR2(1,q);
end


%%%%%%%%%%%%%
% Mode selection and mode coding
%%%%%%%%%%%%%
%intra + copy mode
num_replBocks = zeros(1,n_frames,length(step));
num_replBocks(:,1,:) = 0;  % cannot copy the first frame
%intra + copy mode + moution mode
num_blocks_inmode = zeros(length(step),3);
num_blocks_inmode(:,1) = (video_height/bSize)*(video_width/bSize); %first frame can be only coded
%for each frame
for f=2:n_frames
    %for each quantization step
    for q=1:length(step)
        ww = 1:bSize;
        hh = 1:bSize;
        block_num = 1; %block count for given quntization level and frame
        shift_direction = bestShift(RecoVideo3(:,:,f-1,q),...
            frames(:,:,f),shift_comb,bSize);

        for h = 1:video_height/bSize
            for w = 1:video_width/bSize
                R1=0;
                R2=0;
                R3=0;
                %%%
                % Intra mode + copy mode
                %%%
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
                
                %%%
                % Intra mode only
                %%%
                RecoVideo1(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = bCoded;
                video_rate1(f,q) = video_rate1(f,q) + R1 - 1;
                
                %%%
                % Intra mode + copy mode + moution mode
                %%%
                % Compute motion compensated coordinates
                dy = shift_comb(1,shift_direction(block_num));
                dx = shift_comb(2,shift_direction(block_num));
                y_moved = bSize*(h-1) + dy + hh;
                x_moved = bSize*(w-1) + dx + ww;
                movedBock = RecoVideo3(y_moved,x_moved,f-1,q);
                [recoBlock, R3] = residualCoding(movedBock,frames(y_moved,x_moved,f),q);
                
                Dist3 = distortion(recoBlock,frames(bSize*(h-1)+hh,bSize*(w-1)+ww,f));
                
                R2 = 2; %copy two bits (becouse 3 modes)      
                R1 = R1 + 1;    %two aditional bit for mode selection
                R3 = 2 + 10 + R3;   %two additional bits for mode, 10 bits for vector coding + residual coding.
                % Choose mode that minimizes Lagrangian cost
                Cost1 = Dist1 + lambda11(q)*R1; 
                Cost2 = Dist2 + lambda22(q)*R2;
                Cost3 = Dist3 + lambda33(q)*R3;
                Costf = [Cost1 Cost2 Cost3];
                [MinCost,ChosenMode] = min(Costf);
                
                % Encode video (decide intramode or copy mode)
                if ChosenMode == 1    %intra mode - set to Reco1
                    RecoVideo3(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = bCoded;
                    video_rate3(f,q) = video_rate3(f,q) + R1;
                    num_blocks_inmode(q,1) = num_blocks_inmode(q,1) + 1;
                elseif ChosenMode == 2  %copy block from previous frame
                    RecoVideo3(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = ...
                        RecoVideo3(bSize*(h-1)+hh,bSize*(w-1)+ww,f-1,q);
                    % Save mode selected for visualizations and insight
                    num_blocks_inmode(q,2) = num_blocks_inmode(q,2) + 1;
                    video_rate3(f,q) = video_rate3(f,q) + R2;
                 elseif ChosenMode == 3  %moution compensation mode
                     RecoVideo3(bSize*(h-1)+hh,bSize*(w-1)+ww,f,q) = recoBlock;
                     video_rate3(f,q) = video_rate3(f,q) + R3;
                     num_blocks_inmode(q,3) = num_blocks_inmode(q,3) + 1;
                end  
                
                block_num = block_num + 1;
            end
        end
        PSNR3(f,q) = PSNR(distortion(RecoVideo3(:,:,f,q),frames(:,:,f)));
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
Rate3 = mean(video_rate3,1);
Rate3 = (Rate3*FPS)/1000;
PSNR3_video = mean(PSNR3,1);
plot(Rate1,PSNR1_video,'o-');
hold on;
plot(Rate2,PSNR2_video,'o-');
plot(Rate3,PSNR3_video,'o-');
hold off;
grid on;
title('PSNR vs bit rate for conditional ceplenishment video coder');
legend('intra mode','copy mode', 'moution mode');
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


figure;
bar(step,num_blocks_inmode);
legend('number of intra-frame coded blocks',...
    'number of blocks coded with conditional replacement','number of blocks with motion compensation coding');
title('Number of copied blockes for 3 different modes: intra-coding, conditional replacement and motion compensation');
xlabel('Quzantization step');
ylabel('Number of 16x16 Blocks');

%%
% visualise
%implay(uint8(RecoVideo3(:,:,:,4)),FPS);