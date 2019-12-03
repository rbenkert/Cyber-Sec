import kalman_filter as kf
import numpy as np
import matplotlib.pyplot as plt
import pykalman as pyk
import  kalman_filter as kf

def create_estimate(pos, times_at_pos, velocityup_insteps, velocitydown_insteps, beginn, length, epsilon=0.06):
    res = np.array([])
    for i in range(beginn):
        res = np.append(res, 0)

    count = 0
    cur_index = 0
    cur = pos[cur_index]
    old_pos = 0
    new_pos = 0
    vel_up = velocityup_insteps
    vel_down = velocitydown_insteps
    for i in range(beginn-1, length-1):

        new_pos = pos[cur_index]
        count += 1
        res = np.append(res, new_pos)


        if count == times_at_pos[cur_index]:
            count = 0
            cur_index += 1
            if cur_index == pos.shape[0]: cur_index = 0

    return res


def create_estimate_with_vel(pos, times_at_pos, velocityup_insteps, velocitydown_insteps, beginn, length, epsilon=0.06):
    res = np.array([])
    for i in range(beginn):
        res = np.append(res, 0)

    count = 0
    cur_index = 0
    cur = pos[cur_index]
    old_pos = 0
    new_pos = 0
    vel_up = velocityup_insteps
    vel_down = velocitydown_insteps
    for i in range(beginn - 1, length - 1):

        # if abs(old_pos - pos[cur_index]) < epsilon: count += 1
        if abs(old_pos - pos[cur_index]) < 0.2:
            count += 1
            vel_up = vel_up / 2
            vel_down = vel_down / 2
        if old_pos > pos[cur_index]:
            new_pos = old_pos - vel_down
        else:
            new_pos = old_pos + vel_up
        old_pos = new_pos
        res = np.append(res, new_pos)

        if count == times_at_pos[cur_index]:
            vel_up = velocityup_insteps
            vel_down = velocitydown_insteps
            count = 0
            cur_index += 1
            if cur_index == pos.shape[0]: cur_index = 0

    return res

def create_average_pos(track, beginn, cur_times):
    res = np.array([])
    first = beginn
    for i in range(cur_times.shape[0]):
        average = np.sum(track[first:first+cur_times[i]])/cur_times[i]
        first = first + cur_times[i]
        res = np.append(res, average)
    return res



def simualate_multiple_runs(iter, start_pos, start_times, transition_matr, obs_matr, velocityup_insteps, velocitydown_insteps, beginn, gen=False, track_path='track_data/NormalScenario/track', state_noise=0.02, measurement_noise=0.3, time_step=250, path='Data/PickAndPlace/NormalScenario/'):
    #metaparameters
    show = True
    res_arrayX = np.empty(1)
    res_arrayY = np.empty(1)

    #initialize velocity measurements and kalman filter
    X_trmatr = [[1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
    X_obsmatr = [[1, 0, 0, 0],
                 [0, 1, 0, 0]]
    Y_trmatr = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    Y_obsmatr = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]

    #Init start positions
    ypos_measurements = np.array([start_pos[1][:], start_pos[1][:], start_pos[1][:]])
    ypos_states = np.array([start_pos[1][:], start_pos[1][:], start_pos[1][:]])
    ypos_kf = pyk.KalmanFilter(Y_trmatr, Y_obsmatr, initial_state_mean=[start_pos[1][0], start_pos[1][1], start_pos[1][2], start_pos[1][3], start_pos[1][4], start_pos[1][5], 0, 0, 0, 0, 0, 0])
    ypos_kf.em(ypos_measurements)
    Yfiltered_mean, Yfiltered_covar = ypos_kf.filter(ypos_measurements)

    xpos_measurements = np.array([start_pos[0][:], start_pos[0][:], start_pos[0][:]])
    xpos_states = np.array([start_pos[0][:], start_pos[0][:], start_pos[0][:]])
    xpos_kf = pyk.KalmanFilter(X_trmatr, X_obsmatr,
                               initial_state_mean=[start_pos[0][0], start_pos[0][1], 0, 0])
    xpos_kf.em(xpos_measurements)
    Xfiltered_mean, Xfiltered_covar = xpos_kf.filter(xpos_measurements)



    for t in range(iter):
        x_path = path + 'XPosition' + str(t+1) + '.txt'
        y_path = path + 'YPosition' + str(t+1) + '.txt'
        z_path = path + 'ZPosition' + str(t+1) + '.txt'
        t_path = path + 'Time' + str(t+1) + '.txt'
        xpos = np.loadtxt(x_path)
        ypos = np.loadtxt(y_path)
        #zpos = np.loadtxt(z_path)
        time = np.loadtxt(t_path)
        length = time.shape[0]

        Xmeasurements = np.reshape(xpos ,(xpos.shape[0], -1))
        Ymeasurements = np.reshape(ypos ,(ypos.shape[0], -1))


        #our estimate of the ground truth
        if t<4:
            cur_X = np.array(start_pos[0][:])
            cur_Y = np.array(start_pos[1][:])
            #cur_X_time = np.array(start_times[0][:])
            #cur_Y_time = np.array(start_times[1][:])
        else:
            Xfiltered_mean, Xfiltered_covar = xpos_kf.filter(xpos_measurements)
            Yfiltered_mean, Yfiltered_covar = ypos_kf.filter(ypos_measurements)
            #Xtfiltered_mean, Xtfiltered_covar = xtime_kf.filter(xtime_measurements)
            #Ytfiltered_mean, Ytfiltered_covar = ytime_kf.filter(ytime_measurements)
            cur_X = next_meanx[0:2]
            #cur_X_time = next_meanxt[0:2]
            cur_Y = next_meany[0:6]
            #cur_Y_time = next_meanyt[0:6]

        cur_X_time = np.array(start_times[0][:])
        cur_Y_time = np.array(start_times[1][:])
        #print('Current velocity used for estimate:  %f' %cur_velocity)
        our_statesX = create_estimate(cur_X, cur_X_time, velocityup_insteps, velocitydown_insteps, beginn, length)
        our_statesY = create_estimate(cur_Y, cur_Y_time, velocityup_insteps, velocitydown_insteps, beginn, length)

        #generate track
        print('Generating Track...')
        if gen:
            #trackx = kf.create_track_kf(measurements, transition_matr, obs_matr)
            velocities = trackx[:,1]
            estim_states = trackx[:,0]
        else:
            trackx = np.load(track_path + 'x' + str(t+1) + '.npy')
            tracky = np.load(track_path + 'y' + str(t+1) + '.npy')
            #trackx = np.load(track_path + 'x' + str(t) + '.npy')

        #average vel
        average_xpos = create_average_pos(trackx, beginn, cur_X_time)
        average_ypos = create_average_pos(tracky, beginn, cur_Y_time)
        print('X pos Array:')
        print(average_xpos)
        print('Y pos Array:')
        print(average_ypos)
        if t>= 3:
            ypos_measurements = np.vstack([ypos_measurements, average_ypos])
            next_meany, next_covariancey = ypos_kf.filter_update(Yfiltered_mean[-1], Yfiltered_covar[-1], ypos_measurements[t])
            ypos_states = np.vstack([ypos_states, next_meany[0:6]])

            xpos_measurements = np.vstack([xpos_measurements, average_xpos])
            next_meanx, next_covariancex = xpos_kf.filter_update(Xfiltered_mean[-1], Xfiltered_covar[-1], xpos_measurements[t])
            xpos_states = np.vstack([xpos_states, next_meanx[0:2]])


        #logging params
        residualsX = np.square(our_statesX-trackx[:, 0])
        residualX = np.amax(residualsX)/np.average(residualsX)

        residualsY = np.square(our_statesY-tracky[:, 0])
        residualY = np.amax(residualsY)/np.average(residualsY)

        res_arrayX = np.append(res_arrayX, residualX)
        res_arrayY = np.append(res_arrayY, residualY)
        print('Finished iteration %d!' %t)
        print('Max ResidualX:              %f' %residualX)
        print('Max ResidualY:              %f' %residualY)

        #showing possibility
        if show:
            t = np.arange(0.0, time.shape[0]*time_step, time_step)
            meas = False
            ours = True
            kalman = True
            #ground truth only works when sim is on
            ground_truth = False
            plt.figure(1)
            if meas: plt.scatter(t, ypos, color='red')
            if ours: plt.plot(t, our_statesY[0:t.shape[0]], color='purple', label='Prediction')
            if kalman: plt.plot(t, tracky[0:t.shape[0], 0], color='green', label='Track')
            plt.legend(loc='upper left')
            plt.xlabel('Time')
            plt.ylabel('Y Position')


            plt.figure(2)
            if meas: plt.scatter(t, xpos, color='red')
            if ours: plt.plot(t, our_statesX[0:t.shape[0]], color='purple', label='Prediction')
            if kalman: plt.plot(t, trackx[0:t.shape[0], 0], color='green', label='Track')
            plt.legend(loc='upper left')
            plt.xlabel('Time')
            plt.ylabel('X Position')
            plt.show()
    plt.figure(1)
    t = np.arange(0, iter, 1)
    plt.plot(t, ypos_states[:, 0], color='green')
    plt.scatter(t, ypos_measurements[:, 0], color='red')

    plt.figure(2)
    t = np.arange(0, iter, 1)
    plt.plot(t, xpos_states[:, 0], color='green')
    plt.scatter(t, xpos_measurements[:, 0], color='red')
    #plt.plot(t, vel_hist)
    plt.figure(3)
    t = np.arange(0, iter+1, 1)
    plt.plot(t, res_arrayX)

    plt.figure(4)
    t = np.arange(0, iter+1, 1)
    plt.plot(t, res_arrayY)
    plt.show()


'''
#init start times
    ytime_measurements = np.array([start_times[1][:]])
    ytime_states = np.array([start_times[1][:], start_times[1][:], start_times[1][:]])
    ytime_kf = pyk.KalmanFilter(Y_trmatr, Y_obsmatr,
                               initial_state_mean=[start_times[1][0], start_times[1][1], start_times[1][2], start_times[1][3],
                                                   start_times[1][4], start_times[1][5], 0, 0, 0, 0, 0, 0])
    ytime_kf.em(ytime_measurements)
    Ytfiltered_mean, Ytfiltered_covar = ytime_kf.filter(ytime_measurements)

    xtime_measurements = np.array([start_times[0][:]])
    xtime_states = np.array([start_times[0][:], start_times[0][:], start_times[0][:]])
    xtime_kf = pyk.KalmanFilter(X_trmatr, X_obsmatr,
                               initial_state_mean=[start_times[0][0], start_times[1][1], 0, 0])
    xtime_kf.em(xpos_measurements)
    Xtfiltered_mean, Xtfiltered_covar = xtime_kf.filter(xtime_measurements)
'''