function OUT = comp_idct(in_img, dct_mask)
    [r_in, c_in] = size(in_img);
    M = dct_mask;
    block_x = floor(c_in/M);
    block_y = floor(r_in/M);
    
    A = dctmtx(8);
    OUT = zeros(block_y*M,block_x*M);
    
    for i=1:M:(block_y*M)  %i #row
        for j=1:M:(block_x*M)   %j #cols
            OUT(i:i+(M-1),j:j+(M-1)) = A'*in_img(i:i+(M-1),j:j+(M-1))*A;
        end
    end
end

