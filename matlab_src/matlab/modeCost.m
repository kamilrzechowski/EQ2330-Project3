function [mode_cost,Reco2,Rates] = modeCost(frames,step,video_height,video_width, n_frames,bSize, n_blocks, lambda)
%function for calculating cost of mode 1 and mode 2

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
PSNR2 = zeros(n_frames,length(step));
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
R1 = mean(Rate1,1)*bSize^2 + 1;        %intra mode bitrate
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

%(frame number, quantization level, block number h, block number w, cost mode 1/cost mode 2)
%mode 1 intra, mode 2 copy
mode_cost = zeros(length(frames),length(step),video_height/bSize,video_width/bSize,2);
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
                mode_cost(f,q,h,w,1) = Cost1;
                mode_cost(f,q,h,w,2) = Cost2;
                
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
end

