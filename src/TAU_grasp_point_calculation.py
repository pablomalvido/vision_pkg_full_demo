def get_best_grasping_point(all_cables_input, index_upper, dist_per_pixel, img):
    """
    all_cables_input: list with the index of the cable and its points
    indes_upper: index of the lower upper cable (index 0 is the cable in the bottom)
    """
    upper_cables_all = []
    lower_cables_all = []
    #i=0
    for index in all_cables_input:
        if index >= index_upper:
            upper_cables_all.append(all_cables_input[index])
        else:
            lower_cables_all.append(all_cables_input[index])
        #i+=1

    all_cables = []
    upper_cables = []
    n_top = len(upper_cables_all)
    n_top_wrong = 0
    n_low = len(lower_cables_all)
    n_low_wrong = 0
    wrong = False
    for cable in upper_cables_all: #If a cable doesn't reach the limit complete it with an horizontal value from that point
        if len(cable)>0:
            upper_cables.append(cable)
            all_cables.append(cable)
        else:
            n_top_wrong += 1

    lower_cables = []
    for cable in lower_cables_all:
        if len(cable)>0:
            lower_cables.append(cable)
            all_cables.append(cable)
        else:
            n_low_wrong += 1
 
    xmax = []
    xmin = []

    for cable in all_cables:
        xmax.append(max(p[1] for p in cable))
        xmin.append(min(p[1] for p in cable))
    x_limits = [max(xmin), max(xmax)]

    if min(xmin) > (5/dist_per_pixel) and max(xmax) < img.shape[0] * (3/4):
        wrong = True

    #Complete the cables if there are no more points
    upper_cables_completed = []
    for cable in upper_cables:
        if cable[-1][1] < x_limits[1]:
            if cable[-1][1] < img.shape[0]/2:
                n_top_wrong += 1
            for new_x in range(cable[-1][1]+1, x_limits[1]+1):
                cable.append([cable[-1][0], new_x]) #Keep the last value till the end
        upper_cables_completed.append(cable)

    lower_cables_completed = []
    for cable in lower_cables:
        if cable[-1][1] < x_limits[1]:
            if cable[-1][1] < img.shape[0]/2:
                n_low_wrong += 1
            for new_x in range(cable[-1][1]+1, x_limits[1]+1):
                cable.append([cable[-1][0], new_x]) #Keep the last value till the end
        lower_cables_completed.append(cable)

    if n_top_wrong > n_top/2 or n_low_wrong > n_low/2:
        wrong = True
        print(n_top_wrong)
        print(n_low_wrong)

    cables_up_dict = []
    for cable_up in upper_cables_completed:
        dict_temp = {}
        for point_i in cable_up:
            #print(point_i)
            dict_temp[point_i[1]] = point_i[0]
        cables_up_dict.append(dict_temp)
    
    cables_down_dict = []
    for cable_down in lower_cables_completed:
        dict_temp = {}
        for point_i in cable_down:
            dict_temp[point_i[1]] = point_i[0]
        cables_down_dict.append(dict_temp)

    lower_up_cable_dict = {}
    upper_down_cable_dict = {}
    min_dist_cable_dict = {}
    for x in range(x_limits[0], x_limits[1]+1, 1): #Do it in all the length, not in these limits
        lower_up_x = 0 #y axis goes down
        for cable_up_dict_i in cables_up_dict:
            if cable_up_dict_i[x] > lower_up_x:
                lower_up_x = cable_up_dict_i[x]
        lower_up_cable_dict[x] = lower_up_x

        upper_down_x = 10000
        for cable_down_dict_i in cables_down_dict:
            if cable_down_dict_i[x] < upper_down_x:
                upper_down_x = cable_down_dict_i[x]
        upper_down_cable_dict[x] = upper_down_x

        min_dist_cable_dict[x] = upper_down_cable_dict[x] - lower_up_cable_dict[x]

    best_x = max(min_dist_cable_dict, key=min_dist_cable_dict.get)
    best_y = int((upper_down_cable_dict[best_x] + lower_up_cable_dict[best_x])/2)

    if min_dist_cable_dict[best_x] > 0 and not wrong:
        success_grasp = True
    else:
        success_grasp = False

    return [best_x, best_y], success_grasp