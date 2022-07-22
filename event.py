import numpy as np
import pandas as pd


def create_first_events(simulation_time, event_list, vh_dict, begin_times, names):

    vn = 0
    for vehicle in vh_dict.values():
        vehicle.stage = 0
        first_activity = vehicle.get_current_activity(simulation_time)
        first_event = np.array([begin_times[vn],
                                first_activity[1],
                                begin_times[vn]+first_activity[1],
                                0,
                                first_activity[3],
                                vehicle.vid,
                                vehicle.toid])
        event_list.append(first_event)
        vn = vn + 1

    # sort event list
    event_list = np.array(event_list)
    event_list[:, :4] = event_list[:, :4].astype(float)
    event_list = event_list[event_list[:, names['Begin']].argsort()]

    return event_list, vh_dict


def process_idle(simulation_time, vehicle, current_event, names):
    # begin state
    if current_event[names['State']] == 0:

        # create next event to add to the event list
        next_event = np.array([current_event[names['End']],
                               0.0,
                               current_event[names['End']],
                               1.0,
                               current_event[names['Event']],
                               vehicle.vid,
                               vehicle.toid])
    # end state
    else:
        # find the next relevant Start (for the same vehicle)
        next_activity = vehicle.get_next_activity(simulation_time)
        # change vehicle state
        vehicle.increment_stage()

        # create next event to add to event list
        if next_activity is not None:
            next_event = np.array([next_activity[names['Begin']],
                                   next_activity[names['Duration']],
                                   next_activity[names['End']],
                                   0.0,
                                   next_activity[3],
                                   vehicle.vid,
                                   vehicle.toid])

    return next_event, vehicle


def process_queue(simulation_time, vehicle, current_event, to_dict, queues_to_list, names):

    # begin state
    if current_event[names['State']] == 0:

        # find the next idle TO (if any)
        teleoperator = next((to for to in to_dict.values() if to.status == 'Idle'), None)

        # when there is a TO available
        if teleoperator is not None:

            # assign TO to vehicle
            teleoperator.status = 'Busy'
            teleoperator.vid = vehicle.vid
            vehicle.toid = teleoperator.toid

            # create next (end of queue) event for the vehicle to add to the event list
            next_event = np.array([simulation_time,
                                   0,
                                   simulation_time,
                                   1.0,
                                   current_event[names['Event']],
                                   vehicle.vid,
                                   vehicle.toid])

        # when no TO is available
        else:
            # add vehicle to queue
            queues_to_list.append(vehicle.vid)
            vehicle.q_begin = simulation_time
            next_event = None

    # end state
    else:
        # find next activity for the vehicle (which should be TO takeover)
        next_activity = vehicle.get_next_activity(simulation_time)

        # change vehicle state and assign TO
        vehicle.increment_stage()
        teleoperator = to_dict[vehicle.toid]

        # create next (TO takeover) event for the vehicle to add to the event list
        next_event = np.array([next_activity[names['Begin']],
                               next_activity[names['Duration']],
                               next_activity[names['End']],
                               0.0,
                               next_activity[3],
                               vehicle.vid,
                               vehicle.toid])

    return next_event, vehicle, teleoperator, queues_to_list


def process_takeover(simulation_time, vehicle, current_event, names, takeover_time):

    # begin state
    if current_event[names['State']] == 0:

        # create the end of takeover event
        next_event = np.array([current_event[names['End']],
                               0.0,
                               current_event[names['End']],
                               1.0,
                               current_event[names['Event']],
                               current_event[names['Vehicle']],
                               current_event[names['TO']]])
    # end state
    else:
        # find next activity for the vehicle (which should be TO takeover)
        next_activity = vehicle.get_next_activity(simulation_time)

        # change vehicle state
        vehicle.increment_stage()

        # create next (teleoperation) event for the vehicle to add to the event list
        next_event = np.array([next_activity[names['Begin']],
                               next_activity[names['Duration']],
                               next_activity[names['End']],
                               0.0,
                               next_activity[3],
                               vehicle.vid,
                               vehicle.toid])

    return next_event, vehicle


def process_teleoperated(simulation_time, current_event, names, vehicle, teleoperator, queues_to_list, vh_dict,
                         rest_long, rest_short, max_to_duration):
    # begin state
    if current_event[names['State']] == 0:

        # create the end of teleoperation event
        next_event = np.array([current_event[names['End']],
                               0.0,
                               current_event[names['End']],
                               1.0,
                               current_event[names['Event']],
                               current_event[names['Vehicle']],
                               current_event[names['TO']]])
        next_event_to = None

    # end state
    else:

        # release TO
        vehicle.toid = None
        teleoperator.vid = None
        teleoperator.status = 'Resting'

        # find next activity for the vehicle
        next_activity = vehicle.get_next_activity(simulation_time)

        # create next event for the vehicle to add to the event list
        next_event = np.array([next_activity[names['Begin']],
                               next_activity[names['Duration']],
                               next_activity[names['End']],
                               0.0,
                               next_activity[3],
                               vehicle.vid,
                               vehicle.toid])

        # move vehicle to the next task
        vehicle.increment_stage()

        ## add TO rest activity
        # define rest duration based on teleoperation len
        rest_duration = rest_long if current_event[names['Duration']] > max_to_duration else rest_short
        rest_duration = float(rest_duration)

        # add teleoperator resting event
        next_event_to = np.array([simulation_time,
                                  rest_duration,
                                  simulation_time + rest_duration,
                                  0.0,
                                  'Resting',
                                  None,
                                  teleoperator.toid])

    return next_event, vehicle, teleoperator, queues_to_list, vh_dict, next_event_to


def process_resting(simulation_time, teleoperator, current_event, names, queues_to_list, vh_dict):

    # begin state
    if current_event[names['State']] == 0:

        # create end rest event event to add to the event list
        next_event = np.array([current_event[names['End']],
                               0.0,
                               current_event[names['End']],
                               1.0,
                               current_event[names['Event']],
                               current_event[names['Vehicle']],
                               current_event[names['TO']]])

    # end state
    else:

        # if there was a queue: create next event for the first vehicle in queue
        if queues_to_list:

            # find first vehicle in queue
            next_vehicle = vh_dict[queues_to_list[0]]
            # remove it from the queue
            queues_to_list = queues_to_list[1:]
            # calculate queue duration
            next_vehicle.q_times.append(simulation_time - next_vehicle.q_begin)
            next_vehicle.q_begin = []

            # assign TO to vehicle
            teleoperator.status = 'Busy'
            teleoperator.vid = next_vehicle.vid
            next_vehicle.toid = teleoperator.toid

            # create next event to add to the event list
            next_event = np.array([simulation_time,
                                   next_vehicle.q_times[-1],
                                   simulation_time,
                                   1.0,
                                   'TO Queue',
                                   next_vehicle.vid,
                                   teleoperator.toid])

        else:
            teleoperator.status = 'Idle'
            next_event = None

    return next_event, teleoperator, queues_to_list


def update_event_list(event_list, event_log, next_event, next_event_to, names):

    # eliminate done event & add next event to event list
    if next_event is not None and next_event[names['Event']] != 'Signed off':
        event_list[0] = next_event
        event_log = np.concatenate((event_log, [next_event]))
    else:
        event_list = event_list[1:]

    # add teleoperator resting event (end) to event list
    if next_event_to is not None:
        event_list = np.concatenate((event_list, [next_event_to]))
        event_log = np.concatenate((event_log, [next_event_to]))
        next_event_to = None

    # sort event list
    event_list[:, :4] = event_list[:, :4].astype(float)
    event_list = event_list[event_list[:, names['Begin']].argsort(kind='mergesort')]

    return event_list, event_log, next_event, next_event_to

