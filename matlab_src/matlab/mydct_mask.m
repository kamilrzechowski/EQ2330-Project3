function dct_mask = mydct_mask(size)
    dct_mask = zeros(size);

    for x=0:size-1
        for y=0:size-1
            if x==0
                dct_mask(x+1,y+1) = sqrt(1/size)*cos(((2*y +1)*x*pi)/(2*size));
            else
                dct_mask(x+1,y+1) = sqrt(2/size)*cos(((2*y +1)*x*pi)/(2*size));
            end
        end
    end
end


