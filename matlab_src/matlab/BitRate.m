function rate = BitRate(q_video_frame, step)
    video_width = size(q_video_frame,2); 
    video_height = size(q_video_frame,1); 
    coefs = zeros(8,8);
    H = zeros(video_width/8,video_height/8);
    
    for w=1:(video_width/8)
        for h=1:(video_height/8)
            for ww=1:8
                for hh=1:8
                    coefs(ww,hh) = q_video_frame(8*(h-1)+hh,8*(w-1)+ww);
                end
            end
            %vectors of DCT coefficients
            vals = reshape(coefs(:,:),[1,size(coefs(:,:),1)*size(coefs(:,:),2)]);
            % compute bins to estimate pdfs
            bins_coefs = [min(vals):step:max(vals)];
            if(length(bins_coefs) == 1)
                H(w,h) = -sum(1*log2(1+eps));
            else
                % histogram with bins to get pdfs
                pdf8x8 = hist(vals,bins_coefs)/length(vals);
                % compute entropy from pdfs
                H(w,h) = -sum(pdf8x8.*log2(pdf8x8+eps));
            end
        end
    end
    
    rate = mean2(H);
end

