%projct 3 point 4 Moution
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
block_size = 16;
n_blocks = video_width*video_height/(block_size^2);

% Lagrange multiplier for optimization in choice of coding mode
% To be tuned so that DistortionCOst roughly == RateCost
lambda = 0.02*(step.^2);

%compute costs for mode 1 and 2 and get output using mode 1 and 2
[mode_cost,reco_frames, bits_per_bock] = modeCost(frames,step,video_height,video_width,50,block_size, n_blocks, lambda);
%test passed
%tmp1 = mode_cost(1,1,1,1,1);
%tmp2 = mode_cost(1,1,1,1,2);

%%

%%%%
% Motion mode
%%%%
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

%intaframe coding
framesq = zeros(video_height, video_width,n_frames,length(step));
for f=1:n_frames
    %Intra frame only (mode 1)
    DCT_frame = comp_dct(frames(:,:,f),8);
    for q=1:length(step)
        framesq(:,:,f,q) = comp_idct(quantizer(DCT_frame,step(q)),8);
    end
end

%Do 0 padding
framesPaddedq = zeros(video_height+2*max_shift,...
                   video_width+2*max_shift,n_frames,length(step));
framesPadded = zeros(video_height+2*max_shift,...
    video_width+2*max_shift,n_frames);
for f=1:n_frames
    for q=1:length(step)
        % Pad with zeros
        framesPaddedq(:,:,f,q) = ...
            padarray(framesq(:,:,f,q),[max_shift max_shift]);
    end
    framesPadded(:,:,f) = ...
            padarray(frames(:,:,f),[max_shift max_shift]);
end

%compute model 3
reco_move_framesq = zeros(video_height, video_width,n_frames,length(step));
reco_move_residualq = zeros(video_height, video_width,n_frames,length(step));
reco_mov_frames = zeros(video_height, video_width,n_frames,length(step));
PSNR_reco_mov = zeros(n_frames,length(step));
residual_rate = zeros(n_frames,length(step));
residauls_move_framesDCT = zeros(video_height, video_width,n_frames);
all_used_move_vec = zeros(n_frames,2,n_blocks);
for q=1:length(step)
        % Pad with zeros around to handle the borders in the motion_vec_search
        reco_move_framesq(:,:,1,q) = framesq(:,:,1,q);
end
for f = 1:n_frames - 1
    % find inexes for shift_comb array, whereo shift to minimise MSE
    %find best shift direction
    shift_direction = find_shift_direction(framesPadded(:,:,f),...
        framesPadded(:,:,f+1), shift_comb, block_size,video_width, video_height);
    
    %find residuals
    reco_frame = predict_frame(framesPadded(:,:,f),shift_comb(:,shift_direction),...
            block_size, max_shift, video_width, video_height);
    residauls_move_framesDCT(:,:,f) = comp_dct(frames(:,:,f) - reco_frame,8);
    %save mov vector for entropy coding computation
    all_used_move_vec(f,:,:) = shift_comb(:,shift_direction);
    
    for q = 1:length(step)
        reco_move_framesq(:,:,f+1,q) = predict_frame(framesPaddedq(:,:,f,q),shift_comb(:,shift_direction),...
            block_size, max_shift, video_width, video_height);
        residual_q = quantizer(residauls_move_framesDCT(:,:,f),step(q));
        reco_move_residualq(:,:,f+1,q) = comp_idct(residual_q,8);
        
        reco_mov_frames(:,:,f+1,q) = reco_move_framesq(:,:,f+1,q) + reco_move_residualq(:,:,f+1,q);
        % Bitrate for model 3
        residual_rate(f+1,q) = BitRate(reco_move_residualq(:,:,f+1,q),step(q));
        %compute error
        reco_frame_MSE = distortion(framesq(:,:,f+1,q),reco_mov_frames(:,:,f+1,q));
        PSNR_reco_mov(f+1,q) = PSNR(reco_frame_MSE);
    end

end

% Estimate average bitrate needed for all motion vectors
vals = reshape(all_used_move_vec(:,:,:),...
    [1,size(all_used_move_vec(:,:,:),1)*size(all_used_move_vec(:,:,:),2)*size(all_used_move_vec(:,:,:),3)]);
% compute bins to estimate pdfs
bins_coefs = [min(vals):1:max(vals)];
% histogram with bins to get pdfs
pdf = hist(vals,bins_coefs)/length(vals);
% compute entropy from pdfs
H = -sum(pdf.*log2(pdf+eps));

%total bit rate of model 3
total_mode3_bitrate = zeros(n_frames,length(step));
for f = 2:n_frames
    for q = 1:length(step)
        total_mode3_bitrate(f,q) = residual_rate(f,q) + H;
    end
end
for q = 1:length(step)
    total_mode3_bitrate(1,q) = BitRate(framesq(:,:,1,q), step(q)) + H;
    reco_mov_frames(:,:,1,q) = framesq(:,:,1,q);
end

%combine moe 1,2 and 3
encode_video = zeros(video_height,video_width,n_frames,length(step));
total_bits = zeros(n_frames,length(step));
num_blocks_inmode = zeros(length(step),3);
for f=1:n_frames-1
    for q=1:length(step)
        padded_mode3 = padarray(reco_mov_frames(:,:,f,q),[max_shift max_shift]);
            
        ww = 1:block_size; %indices to get 16x16 blocks
        hh = 1:block_size;
        block_count = 1;
        for h = 1:video_height/block_size
            for w = 1:video_width/block_size
                % Compute motion compensated coordinates
                dy = all_used_move_vec(f,1,block_count);
                dx = all_used_move_vec(f,2,block_count);
                y_moved = block_size*(h-1) + dy + hh + max_shift;
                x_moved = block_size*(w-1) + dx + ww + max_shift;
                diff = (frames(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1) - ...
                    padded_mode3(y_moved,x_moved)).^2;
                diff_mode3 = sum(diff(:))/numel(diff(:));
                
                Cost1 = mode_cost(f,q,h,w,1);
                Cost2 = mode_cost(f,q,h,w,2);
                Cost3 = diff_mode3 + lambda(q)*(total_mode3_bitrate(q)/n_blocks);
                Costv = [Cost1,Cost2,Cost3];
                [MinCost,Chosen_Mode] = min(Costv);
                
                %choose mode for current block
                if(Chosen_Mode == 1)
                    encode_video(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1,q) = ...
                        reco_frames(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1,q);
                    total_bits(f+1,q) = total_bits(f+1,q) + bits_per_bock(1);
                    num_blocks_inmode(q,1) = num_blocks_inmode(q,1) + 1;
                elseif (Chosen_Mode == 2)
                    encode_video(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1,q) = ...
                        reco_frames(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1,q);
                    total_bits(f+1,q) = total_bits(f+1,q) + bits_per_bock(2);
                    num_blocks_inmode(q,2) = num_blocks_inmode(q,2) + 1;
                else
                    encode_video(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1,q) = ...
                        reco_mov_frames(block_size*(h-1)+hh,block_size*(w-1)+ww,f+1,q);
                    total_bits(f+1,q) = total_bits(f+1,q) + (total_mode3_bitrate(f+1,q)/n_blocks);
                    num_blocks_inmode(q,3) = num_blocks_inmode(q,3) + 1;
                end
                
                block_count = block_count+1;
            end
        end
    end
end

%first frame
for q = 1:length(step)
    encode_video(:,:,1,q) = framesq(:,:,1,q);
    total_bits(1,q) = bits_per_bock(1)*n_blocks;
end

%%
% visualise
%implay(uint8(encode_video(:,:,:,3)),FPS);


% plot bit rate vs PSNR
PSNR_frames = zeros(n_frames,length(q));
for f=1:n_frames-1
    for q=1:length(step)
        PSNR_frames(f,q) = PSNR(distortion(frames(:,:,f),encode_video(:,:,f,q)));
    end
end
%bit_rate_per_frame = zeros(length(step),1);
PSNR_video = mean(PSNR_frames,1);
%for q = 1:length(step)
%    for f = 1:n_frames
%        bit_rate_per_frame(q) = bit_rate_per_frame(q) + total_bits(f,q);
%    end
%end
bit_rate_per_frame = mean(total_bits(:,:),1);
figure;
%bit rate in kbps
bit_rate_per_frame = (bit_rate_per_frame*FPS)/1000;
plot(bit_rate_per_frame,PSNR_video, 'linewidth', 2);
title('PSNR vs bit rate for coding with fiuson of 3 modes');
xlabel('Bit rate [kbps]');
ylabel('PSNR');



%plot_mat = [replacedBlock_count(:),totalBock_count(:)];
figure;
bar(step,num_blocks_inmode);
legend('number of intra-frame coded blocks',...
    'number of blocks coded with conditional replacement','number of blocks with motion compensation coding');
title('Number of copied blockes for 3 different modes: intra-coding, conditional replacement and motion compensation');
xlabel('Quzantization step');
ylabel('Number of 16x16 Blocks');