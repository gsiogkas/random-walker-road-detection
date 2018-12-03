function [xyroad,xyoccl,bw] = read_labeled_data(filename)

    f1 = fopen(filename);
    sz = [240 320];

    for i = 1:100,
        c=(fgetl(f1));
        if c == -1, 
            break;
        end
        x(i,:) = c(1:4);
    end
    fclose(f1);
    clear f1
    f1 = fopen(filename);
    k=0;
    for i = 1:size(x,1),
        if sum(x(i,:)=='road')==4
            road =  fscanf(f1,'%s',1);
            nvroad =  fscanf(f1,'%d',1);
            xyroad = zeros(nvroad,2);
            for j = 1:nvroad
                xyroad(j,1) = fscanf(f1,'%f',1);
                xyroad(j,2) = fscanf(f1,'%f',1);    
            end
        elseif sum(x(i,:)=='occl')==4
            occl =  fscanf(f1,'%s',1);
            k = k + 1;
            nvoccl =  fscanf(f1,'%d',1);
            
            for j = 1:nvoccl
                xyoccl(j,1,k) = fscanf(f1,'%f',1);
                xyoccl(j,2,k) = fscanf(f1,'%f',1);
            end
        end
    end
    bwroad = poly2mask(xyroad(:,1).*sz(2),xyroad(:,2).*sz(1),sz(1),sz(2));

    if k > 0,
        bwoccl = poly2mask(xyoccl(:,1,1).*sz(2),xyoccl(:,2,1).*sz(1),sz(1),sz(2));
        for i = 2:k,
            bwoccl = bwoccl | poly2mask(xyoccl(:,1,i).*sz(2),xyoccl(:,2,i).*sz(1),sz(1),sz(2));
        end
        bw = bwroad & ~bwoccl;
    else
        bw = bwroad;
        xyoccl = 0;
    end
    bw(:, end) = bw(:, end-1);
    bw(end, :) = bw(end-1, :);
    fclose(f1);
end