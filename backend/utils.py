def lowpass(input,moving_arr,index,size):
    if(index<size-1):
        index+=1
        moving_arr[index] = input
    elif(index>=size-1):
        index+=1
        index = index%10
        moving_arr[index] = input
    return np.sum(moving_arr)/size


def pause(x):
    '''
    if(DATA(["left_angle"]>10 or DATA["left_speed"]>0.5):
        present_state["left_arm"] = STATES["down"]
    elif(DATA["left_angle"]< 115 or DATA["left_speed"]<-0.1):
        present_state["left_arm"] = STATES["up"]
    '''
    if(x==0):
        if(NORM["left_angle"]>0.55 and DATA["left_speed"]>5):
            present_state["left_arm"] = STATES["down"]
        elif(NORM["left_angle"]<0.90 and DATA["left_speed"]<-5):
            present_state["left_arm"] = STATES["up"]
        return present_state["left_arm"]
    elif(x==1):
        if(NORM["right_angle"]>0.55 and DATA["right_speed"]>5):
            present_state["right_arm"] = STATES["down"]
        elif(NORM["right_angle"]<0.90 and DATA["right_speed"]<-5):
            present_state["right_arm"] = STATES["up"]
        return present_state["right_arm"]

def up(x):
    '''
    if(DATA["left_angle"]<15):
        present_state["left_arm"] = STATES["pause"]
    if(DATA["left_speed"]<-4):
        print("Move your hand slowly")
    if(DATA["left_speed"]>1):
        present_state["left_arm"] = STATES["down"]
        print("Pause State missed...")
    '''
    if(x==0):
        if(NORM["left_angle"]<0.55 ):
            present_state["left_arm"] = STATES["pause"]
        elif(NORM["left_angle"]>0.55 and DATA["left_speed"]>5):
            present_state["left_arm"] = STATES["down"]
            print("pause state missed")
        return present_state["left_arm"]
    elif(x==1):
        if(NORM["right_angle"]<0.55 ):
            present_state["right_arm"] = STATES["pause"]
        elif(NORM["right_angle"]>0.55 and DATA["right_speed"]>5):
            present_state["right_arm"] = STATES["down"]
            print("pause state missed")
        return present_state["right_arm"]

def down(x):
    if(x==0):
        if(NORM["left_angle"]>0.90):
            present_state["left_arm"] = STATES["pause"]
        elif(NORM["left_angle"]<0.90 and DATA["left_speed"]<-5):
            present_state["left_arm"] = STATES["up"]
            print("Pause state missed")
        return present_state["left_arm"]
    elif(x==1):
        if(NORM["right_angle"]>0.90):
            present_state["right_arm"] = STATES["pause"]
        elif(NORM["right_angle"]<0.90 and DATA["right_speed"]<-5):
            present_state["right_arm"] = STATES["up"]
            print("Pause state missed")
        return present_state["right_arm"]

def fsm_arm(x,arm):
    switcher = {0:partial(pause,arm), 1:partial(up,arm), 2:partial(down,arm)}
    func=switcher.get(x)
    return func()

def init_param(MIN, MAX):
    MIN["left_angle"] = 1000
    MIN["left_speed"] = 1000
    MIN["right_angle"] = 1000
    MIN["right_speed"] = 1000

    MAX["left_angle"] = -1000
    MAX["left_speed"] = -1000
    MAX["right_angle"] = -1000
    MAX["right_speed"] = -1000
    ti =0
    tf =0
    #temp1 =0
    #temp2 =0

def update_max():
    if(MAX["left_angle"] <= DATA["left_angle"]):
        MAX["left_angle"] = DATA["left_angle"]

    if(MAX["right_angle"] <= DATA["right_angle"]):
        MAX["right_angle"] = DATA["right_angle"]

    if(MAX["left_speed"] <= DATA["left_speed"]):
        MAX["left_speed"] = DATA["left_speed"]
    
    if(MAX["right_speed"] <= DATA["right_speed"]):
        MAX["right_speed"] = DATA["right_speed"]

def update_min():
    if(MIN["left_angle"] >= DATA["left_angle"]):
        MIN["left_angle"] = DATA["left_angle"]

    if(MIN["right_angle"] >= DATA["right_angle"]):
        MIN["right_angle"] = DATA["right_angle"]

    if(MIN["left_speed"] >= DATA["left_speed"]):
        MIN["left_speed"] = DATA["left_speed"]
    
    if(MIN["right_speed"] >= DATA["right_speed"]):
        MIN["right_speed"] = DATA["right_speed"]
    
def get_angle(arr,num):
    dist1 = 0
    dist2 = 0
    dist3 = 0
    if(num==0):
        x1 = arr[0,PART_NAMES["leftWrist"],0]
        y1 = arr[0,PART_NAMES["leftWrist"],1]
        x2 = arr[0,PART_NAMES["leftElbow"],0]
        y2 = arr[0,PART_NAMES["leftElbow"],1]
        x3 = arr[0,PART_NAMES["leftShoulder"],0]
        y3 = arr[0,PART_NAMES["leftShoulder"],1]
    elif(num==1):
        x1 = arr[0,PART_NAMES["rightWrist"],0]
        y1 = arr[0,PART_NAMES["rightWrist"],1]
        x2 = arr[0,PART_NAMES["rightElbow"],0]
        y2 = arr[0,PART_NAMES["rightElbow"],1]
        x3 = arr[0,PART_NAMES["rightShoulder"],0]
        y3 = arr[0,PART_NAMES["rightShoulder"],1]
                 
    dist1 = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    dist2 = math.sqrt((x2-x3)**2 + (y2-y3)**2)
    dist3 = math.sqrt((x1-x3)**2 + (y1-y3)**2)
    #print(dist3)
    cosine = (dist1**2 + dist2**2 - dist3**2)/(2*dist1*dist2)

    return math.acos(cosine)*(180/math.pi) 


def print_armstate(x,arm):
    if(arm==0):
        if(x==0):
            print("left: pause")
        elif(x==1):
            print("left: up")
        elif(x==2):
            print("left: down")
    elif(arm==1):
        if(x==0):
            print("right: pause")
        elif(x==1):
            print("right: up")
        elif(x==2):
            print("right: down")
'''
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
'''
def check_speed():
    if(DATA["avg_left_speed"]>20):
        print("Move your left hand slowly")
    if(DATA["avg_right_speed"]>20):
        print("MOve your right hand slowly")
