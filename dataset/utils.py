from typing import List, Optional
import cv2


class DrawLanemarks:
    def __init__(self, draw_line: bool = True, draw_grid: Optional[List[int]] = None):
        self.lanetype_colors = [(250, 170, 30),  # 1 road curb
                                (0, 255, 0),     # 2 double white
                                (100, 255, 0),   # 3 double yellow
                                (0, 255, 100),   # 4 double other
                                (255, 0, 0),     # 5 single white
                                (255, 100, 0),   # 6 single yellow
                                (255, 0, 100),   # 7 single other
                                (50, 0, 100),    # 8 crosswalk
                                (0, 0, 0), ]     # 0 unknown
        self.instance_colors = [(250, 0, 0),
                                (0, 255, 0),
                                (0, 0, 255),
                                (255, 255, 0),
                                (0, 255, 255),
                                (255, 0, 255)]
        self.draw_line = draw_line
        self.draw_grid = draw_grid

    def __call__(self, **kwargs):
        img_disp = kwargs["image"] * 1  # copy
        for idx, each_lane in enumerate(kwargs["lane_list"]):
            if "lane_type" in kwargs:  # if lane_type exist, the use lanetype_colors
                type_id = kwargs["lane_type"][idx]
                if self.draw_line is True:
                    cv2.polylines(img_disp, [each_lane.astype(int)], False, self.lanetype_colors[type_id], 2)
                for pt in each_lane:
                    cv2.circle(img_disp, (int(pt[0]), int(pt[1])), 3, self.lanetype_colors[type_id], -1)
            else:
                if self.draw_line is True:
                    cv2.polylines(img_disp, [each_lane.astype(int)], False, self.instance_colors[idx], 2)
                for pt in each_lane:
                    cv2.circle(img_disp, (int(pt[0]), int(pt[1])), 3, self.instance_colors[idx], -1)

        if self.draw_grid is not None:
            rows, cols = img_disp.shape[:2]
            x_step = cols // self.draw_grid[1]
            for x in range(64):
                cv2.line(img_disp, (x * x_step, 0), (x * x_step, rows), (0, 0, 255), 1)
            y_step = rows // self.draw_grid[0]
            for y in range(32):
                cv2.line(img_disp, (0, y * y_step), (cols, y * y_step), (0, 0, 255), 1)
        return img_disp

