function [non_road_seeds, road_seeds] = get_seeds(hsc1, thresh, prev_road)
    
    % Calculating coordinates for road trapezoid 
    % These need to be defined manually, according to the camera placement
    [rows, cols] = size(hsc1);
    top_left = [floor(.7 * rows), floor(.35 * cols)];
    top_right = [floor(.7 * rows), floor(.65 * cols)];
    bottom_left = [rows - 1, floor(.1 * cols)];
    bottom_right = [rows - 1, floor(.9 * cols)];
    road_trapezoid = [rows - 1, top_left(1), top_right(1), rows - 1;... 
                      bottom_left(2), top_left(2), top_right(2), bottom_right(2)];

    % Getting the road seed coordinates
    if ~isempty(prev_road) % Case of no feedback (always same road seeds)
        road_perim = adaptive_perimeter_update(prev_road, hsc1 < thresh);
    else % Case of adaptive road seed generation, using previous result
        road_mask = poly2mask(road_trapezoid(2,:), road_trapezoid(1,:),...
                              rows, cols);
        road_perim = road_mask & hsc1 < thresh; 
    end
    % Converting road seeds image mask to coordinates
    [rows_road, cols_road] = find(road_perim > 0);
    
    % Getting the non-road seed coordinates
    sky = zeros(size(hsc1));
    sky(1:floor(.2 * rows), :) = 1;
    [rows_non_road, cols_non_road] = find(sky | hsc1 >= thresh);
    
    % Converting all coordinates to 1-d indices (for vectorized use of RW)
    non_road_seeds = (sub2ind([rows, cols], rows_non_road, cols_non_road));
    road_seeds = (sub2ind([rows, cols], rows_road, cols_road));
end