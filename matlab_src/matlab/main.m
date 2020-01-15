% Project 3 point 2
clear

%Uniform Quantizer 
step = [2^3 2^4 2^5 2^6];
FPS = 30;
video_width = 176; video_height = 144;
% Import video luminance [video_width, vieo_height] 
Video = yuv_import_y('foreman_qcif.yuv',[video_width video_height],50);

%init variables
err=zeros(length(Video),length(step));


%DCT-2
for i=1:length(Video)
    %compute dct
    Video_dct{i,1} = comp_dct(Video{i,1},8);
    
    %quantize
    for j=1:length(step)
        s = step(j);
        Video_dct_q{i,j} = quantizer(Video_dct{i,1},s);
    
        %compute mean square error
        err(i,j) = distortion(Video_dct{i,1},Video_dct_q{i,j});

        %reconstruct video ,i - frame, j - guntization step
        Video_rec{i,j} = comp_idct(Video_dct_q{i,j},8);
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

%% bitrate estimtion
rates = zeros(size(step));

%calculate bitrates for each qunatizer step 
bSize = 16;
for j=1:length(step)
    s = step(j);
    coefs = zeros(bSize,bSize);
    H = zeros(video_width/bSize,video_height/bSize);
    
    for i=1:length(Video_dct_q) %for each frame
        for w=1:(video_width/bSize)
            for h=1:(video_height/bSize)
                for ww=1:bSize
                    for hh=1:bSize
                        coefs(ww,hh) = Video_dct_q{i,j}(bSize*(h-1)+hh,bSize*(w-1)+ww);
                    end
                end
                %vectors of DCT coefficients
                vals = reshape(coefs(:,:),[1,size(coefs(:,:),1)*size(coefs(:,:),2)]);
                % compute bins to estimate pdfs
                bins_coefs = [min(vals):s:max(vals)];
                % histogram with bins to get pdfs
                pdf8x8 = hist(vals,bins_coefs)/length(vals);
                % compute entropy from pdfs
                H(w,h) = -sum(pdf8x8.*log2(pdf8x8+eps));
            end
        end
    end
    
    rates(j) = mean2(H);
end

%covert to kbit/s
ratesKBPS = rates .* ((video_height*video_width*FPS)/1000);

figure;
plot(ratesKBPS, meanPSNR, 'linewidth', 2);
title('PSNR vs bitrate (DCT)');
grid on;
xlabel('Bit rate [kbps]');
ylabel('PSNR [dB]');
%xlim([0.2 1.6]);