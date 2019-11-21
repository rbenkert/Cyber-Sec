import pykalman as pyk
import numpy as np
import matplotlib.pyplot as plt

def create_velocity_model(initial_height, length, time_step, velocity, noise, state_noise):
    old_state = initial_height
    old_meas = initial_height + noise*np.random.normal()
    states = np.array([[initial_height]])
    measurements = np.array([[old_meas]])
    for t in range(1, length):
        new_vel = velocity+ state_noise*np.random.normal()
        new_state = old_state + new_vel*time_step
        #new_state = old_state + velocity*time_step + state_noise*np.random.normal()
        new_meas = new_state + noise*np.random.normal()
        states = np.vstack([states, new_state])
        measurements = np.vstack([measurements, new_meas])
        old_state = new_state
    return states, measurements

def fill_up_create_model(initial_height, target, time_step, velocity, noise, state_noise):
    old_state = initial_height
    old_meas = initial_height + noise*np.random.normal()
    states = np.array([[initial_height]])
    measurements = np.array([[old_meas]])
    len_on = 0
    while old_state<target:
        new_vel = velocity+ state_noise*np.random.normal()
        new_state = old_state + new_vel*time_step
        #new_state = old_state + velocity*time_step + state_noise*np.random.normal()
        new_meas = new_state + noise*np.random.normal()
        states = np.vstack([states, new_state])
        measurements = np.vstack([measurements, new_meas])
        old_state = new_state
        len_on += 1
    #if len_on != 0: len_on += 1
    return states, measurements, len_on

def create_estimate(initial_height, length, time_step, velocity, state_noise):
    old_state = initial_height
    states = np.array([[initial_height]])
    for t in range(1, length):
        new_vel = velocity + state_noise*np.random.normal()
        new_state = old_state + new_vel*time_step
        #new_state = old_state + velocity*time_step + state_noise*np.random.normal()
        states = np.vstack([states, new_state])
        old_state = new_state
    return states

def create_estimate_fill(initial_height, target, time_step, velocity, state_noise):
    old_state = initial_height
    states = np.array([[initial_height]])
    len_on = 0
    while old_state < target:
        new_vel = velocity + state_noise*np.random.normal()
        new_state = old_state + new_vel*time_step
        #new_state = old_state + velocity*time_step + state_noise*np.random.normal()
        states = np.vstack([states, new_state])
        old_state = new_state
        len_on += 1
    #if len_on != 0: len_on += 1
    return states, len_on



def create_track_kf(measurements, transition_matrix, observation_matrix, init_state=None):
    kf = pyk.KalmanFilter(transition_matrix, observation_matrix)
    estim_measurements = measurements[:3,:]
    kf.em(estim_measurements)
    if init_state != None:
        kf.initial_state_mean = init_state
    estim_state = np.array([kf.initial_state_mean])
    for t in range(1, measurements.shape[0]):
        old_mes = measurements[:t,:]
        filtered_mean, filtered_covar = kf.filter(old_mes)
        next_mean, next_covariance = kf.filter_update(filtered_mean[-1], filtered_covar[-1], measurements[t])
        estim_state = np.vstack([estim_state, [next_mean]])

    return estim_state

def simualate_multiple_runs(iter, start_velocity, change_rate, start_of_change, transition_matr, obs_matr, target=15.0, len_sim=150, time_step=0.2, state_noise=0.02, measurement_noise=0.3):
    #metaparameters
    show = True
    res_array = np.empty(1)

    #init velocity dropping
    vel_hist = np.array([start_velocity])
    for i in range(1,start_of_change):
        vel_hist = np.append(vel_hist, start_velocity)
    for i in range(start_of_change, iter):
        vel_hist = np.append(vel_hist, vel_hist[-1]-change_rate)

    #initialize velocity measurements and kalman filter
    vel_trmatr = [[1, 1], [0, 1]]
    vel_obsmatr = [[1,0]]
    vel_measurements = np.array([[start_velocity], [start_velocity], [start_velocity]])
    vel_states = np.array([start_velocity, start_velocity, start_velocity])
    vel_kf = pyk.KalmanFilter(vel_trmatr, vel_obsmatr, initial_state_mean=[start_velocity, 0])
    vel_kf.em(vel_measurements)
    filtered_mean, filtered_covar = vel_kf.filter(vel_measurements)

    for t in range(iter):
        #ground truth
        true_states, measurements, len_on = fill_up_create_model(0.0, target, time_step, vel_hist[t], measurement_noise, state_noise)
        len_off = len_sim - len_on
        if len_off > 0:
            phase_off, phase_off_meas = create_velocity_model(true_states[-1, 0], len_off, time_step, 0.0, measurement_noise, state_noise)
            true_states = np.vstack([true_states, phase_off])
            measurements = np.vstack([measurements, phase_off_meas])
        else:
            measurements = measurements[0:(len_sim+1),:]

        #our estimate of the ground truth
        if t<4:
            cur_velocity = start_velocity
        else:
            filtered_mean, filtered_covar = vel_kf.filter(vel_measurements)
            cur_velocity = next_mean[0]
        our_states, len_on_ours = create_estimate_fill(0.0, target, time_step, cur_velocity, state_noise)
        len_off_ours = len_sim - len_on_ours
        if len_off_ours > 0:
            phase2_ours = create_estimate(our_states[-1, 0], len_off_ours, time_step, 0, state_noise)
            our_states = np.vstack([our_states, phase2_ours])
        else:
            our_states = our_states[0:(len_sim+1), :]

        #generate track
        track = create_track_kf(measurements, transition_matr, obs_matr)
        velocities = track[:,1]
        estim_states = track[:,0]

        #average vel
        average_velocity = np.sum(velocities[0:len_on_ours])/len_on_ours
        if t>= 3:
            vel_measurements = np.vstack([vel_measurements, [average_velocity]])
            next_mean, next_covariance = vel_kf.filter_update(filtered_mean[-1], filtered_covar[-1], vel_measurements[t])
            vel_states = np.append(vel_states, next_mean[0])


        #logging params
        residuals = np.square(our_states-estim_states)
        residual = np.amax(residuals)
        vel_error = (cur_velocity-vel_hist[i])**2

        res_array = np.append(res_array, residual)
        print('Finished iteration %d!' %t)
        print('Max Residual:              %f' %residual)
        print('Square Error of velocity:  %f' %vel_error)

        #showing possibility
        if show:
            t = np.arange(0.0, (len_sim+1)*time_step, time_step)
            meas = True
            ours = True
            kalman = True
            ground_truth = True
            meas_plot = measurements[:,0]
            meas_plot = np.reshape(meas_plot,-1)
            if meas: plt.scatter(t, meas_plot, color='red')
            if ours: plt.plot(t, our_states[0:(len_sim+1),0], color='purple')
            if kalman: plt.plot(t, track[0:(len_sim+1),0], color='green')
            # plt.plot(t, state, color= 'red')
            if ground_truth: plt.plot(t, true_states[0:(len_sim+1),0])
            plt.show()
    plt.figure(1)
    t = np.arange(0, iter, 1)
    plt.plot(t, vel_states, color='green')
    plt.scatter(t, vel_measurements, color='red')
    plt.plot(t, vel_hist)
    plt.figure(2)
    t = np.arange(0, iter+1, 1)
    plt.plot(t, res_array)
    plt.show()










