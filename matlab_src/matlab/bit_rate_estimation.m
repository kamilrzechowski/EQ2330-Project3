function rates = bit_rate_estimation(Video_dct_q, n_frames,step, video_width, video_height)
    rates = zeros(size(step));
    for q=1:length(step)
        s = step(q);
        coefs = zeros(8,8);
        H = zeros(video_width/8,video_height/8);

        for f=1:n_frames %for each frame
            for w=1:(video_width/8)
                for h=1:(video_height/8)
                    for ww=1:8
                        for hh=1:8
                            coefs(ww,hh) = Video_dct_q(8*(h-1)+hh,8*(w-1)+ww,f,q);
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

        rates(q) = mean2(H);
    end
end

