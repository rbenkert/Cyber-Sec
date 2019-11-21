import kalman_filter as myKalman
import numpy as np
import matplotlib.pyplot as plt






if __name__ == '__main__':
    time_step = 0.1
    len_on = 100
    len_off = 50
    len_sim = 150
    target = 50.0

    #states and measurements
    velocity = 5.0
    measurement_noise = 0.8
    state_transition_noise = 0.06
    start_of_change = 4
    iter = 10
    change_rate = 0.8

    #kf parameters
    transition_matrix = [[1, time_step], [0, 1]]
    observation_matrix = [[1, 0]]

    myKalman.simualate_multiple_runs(iter, velocity, change_rate, start_of_change, transition_matrix, observation_matrix, target, len_sim, time_step, state_transition_noise, measurement_noise)




'''



    states, measurements = myKalman.create_velocity_model(0, 100, time_step, velocity, measurement_noise, state_transition_noise)
    t = np.arange(0.0, len_sim*time_step, time_step)



    #phase2
    new_velocity = 0
    phase2_st, phase2_m = myKalman.create_velocity_model(states[-1,0], 50, time_step, new_velocity, measurement_noise, state_transition_noise)
    states = np.vstack([states, phase2_st])
    measurements = np.vstack([measurements, phase2_m])

    bad_state, bad_meas = myKalman.create_velocity_model(0, len_sim, time_step, velocity, measurement_noise, state_transition_noise)
    state, meas = myKalman.create_velocity_model(0, len_sim, time_step, 0.1, measurement_noise,state_transition_noise)

    # estimated state
    transition_matrix = [[1, time_step], [0, 1]]
    observation_matrix = [[1, 0]]
    track = myKalman.create_track_kf(measurements, transition_matrix, observation_matrix)
    estim_height = track[:, 0]
    print('Estimated Velocity t=140: %f' % track[-1, 1])
    print('Estimated Velocity t=50: %f' % track[50, 1])
    velocity_aver_on = sum(track[0:100, 1])/100
    print('Average Velocity on: %f' % velocity_aver_on)
    velocity_aver = sum(track[100:len_sim, 1])/50
    print('Average Velocity off: %f' % velocity_aver)

    states_estim_vel, measurements_estim_vel = myKalman.create_velocity_model(0, 100, time_step, velocity_aver_on, measurement_noise, state_transition_noise)
    phase2_st_vel, phase2_m_vel = myKalman.create_velocity_model(states_estim_vel[-1,0], 50, time_step, velocity_aver, measurement_noise, state_transition_noise)
    states_estim = np.vstack([states_estim_vel, phase2_st_vel])










    myst = np.reshape(states, -1)
    mymeas = np.reshape(measurements, -1)


    plt.scatter(t, mymeas, color='red')
    plt.plot(t, myst)
    #plt.plot(t, states_estim, color='red')
    #plt.plot(t, state, color= 'red')
    plt.plot(t, estim_height, color='green')
    plt.show()
'''