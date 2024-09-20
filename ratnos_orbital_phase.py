# Load required standard modules
import os
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation, propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.kernel.astro import gravitation
from tudatpy import astro


def get_gravity_enceladus():

    mu_enceladus = 7.211292085479989E+9
    radius_enceladus = 252240.0
    cosine_coef = np.zeros((10, 10))
    sine_coef = np.zeros((10, 10))

    cosine_coef[0, 0] = 1.0

    cosine_coef[2, 0] = 5.4352E-03 / gravitation.legendre_normalization_factor(2, 0) # wrong, correct is -5.4 ...
    cosine_coef[2, 1] = 9.2E-06 / gravitation.legendre_normalization_factor(2, 1)
    cosine_coef[2, 2] = 1.5498E-03 / gravitation.legendre_normalization_factor(2, 2)

    cosine_coef[3, 0] = -1.15E-04 / gravitation.legendre_normalization_factor(3, 0) # wrong, correct is 1.15 ...

    sine_coef[2, 1] = 3.98E-05 / gravitation.legendre_normalization_factor(2, 1)
    sine_coef[2, 2] = 2.26E-05 / gravitation.legendre_normalization_factor(2, 2)

    return environment_setup.gravity_field.spherical_harmonic(mu_enceladus, radius_enceladus, cosine_coef, sine_coef, "IAU_Enceladus")

def getKaulaConstraint(kaula_constraint_multiplier, degree): #
    return kaula_constraint_multiplier / degree ** 2

def apply_kaula_constraint_a_priori(kaula_constraint_multiplier, max_deg_gravity, indices_cosine_coef, indices_sine_coef, inv_apriori):

    index_cosine_coef = indices_cosine_coef[0]
    index_sine_coef = indices_sine_coef[0]

    for deg in range(2, max_deg_gravity + 1):
        kaula_constraint =getKaulaConstraint(kaula_constraint_multiplier, deg)
        for order in range(deg + 1):
            inv_apriori[index_cosine_coef, index_cosine_coef] = kaula_constraint ** -2
            index_cosine_coef += 1
        for order in range(1, deg + 1):
            inv_apriori[index_sine_coef, index_sine_coef] = kaula_constraint ** -2
            index_sine_coef += 1


# Load spice kernels
kernels = ['C:\TudatProjects\OrbitIntegration\de438.bsp', 'C:\TudatProjects\OrbitIntegration\sat427.bsp', 'C:\TudatProjects\OrbitIntegration\pck00010.tpc']
spice.load_standard_kernels(kernels)


# Set simulation start and end epochs
start_gco = 0. * constants.JULIAN_YEAR  # beginning circular orbital phase 35.3844 * constants.JULIAN_YEAR
end_gco = start_gco + 100.0 * constants.JULIAN_DAY # 35.73 * constants.JULIAN_YEAR  # end circular orbital phase

# Define global propagation settings Enceladus
global_frame_origin = "Enceladus"
global_frame_orientation = "J2000"
body_to_propagate = ["Orbiter"]
central_body = ["Enceladus"]

# Create default body settings Enceladus
bodies_to_create = ["Enceladus", "Saturn", "Sun"]
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# Define the spherical harmonics gravity model for Saturn
saturn_gravitational_parameter = 3.7931208E+16
saturn_reference_radius = 60330000.0

# Normalize the spherical harmonic coefficients
nor_sh_sat=astro.gravitation.normalize_spherical_harmonic_coefficients(
    [ #Iess et al. 2019, as in the minimal example by Andreas
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-16290.71E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [935.83E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-86.14E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10.E-6, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [ #Iess et al. 2019, as in the minimal example by Andreas
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

# Assign normalized cosine and sine coefficients
saturn_normalized_cosine_coefficients = nor_sh_sat[0]
saturn_normalized_sine_coefficients = nor_sh_sat[1]

saturn_associated_reference_frame = "IAU_Saturn"

# Create the gravity field settings and add them to the body "Saturn"
body_settings.get( "Saturn" ).gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    saturn_gravitational_parameter,
    saturn_reference_radius,
    saturn_normalized_cosine_coefficients,
    saturn_normalized_sine_coefficients,
    saturn_associated_reference_frame )

# Add setting for moment of inertia for Saturn
body_settings.get("Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210

# Compute rotation rate for Enceladus
mu_saturn = spice.get_body_properties("Saturn", "GM", 1)[0] * 1.0e9
initial_state_enceladus = spice.get_body_cartesian_state_at_epoch("Enceladus", "Saturn", "J2000", "None", start_gco)
keplerian_state_enceladus = element_conversion.cartesian_to_keplerian(initial_state_enceladus, mu_saturn)
rotation_rate_enceladus = np.sqrt(mu_saturn/keplerian_state_enceladus[0]**3)

# Set rotation model settings Enceladus
initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000", "IAU_Enceladus", start_gco)
body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
    "J2000", "IAU_Enceladus", initial_orientation_enceladus, start_gco, rotation_rate_enceladus)

# Gravity field settings
body_settings.get("Enceladus").gravity_field_settings = get_gravity_enceladus()

# Create empty settings for RaTNOS Orbiter
body_settings.add_empty_settings("Orbiter")

# Create empty multi-arc ephemeris for RaTNOS orbiter
empty_ephemeris_dict = dict()
orbiter_ephemeris = environment_setup.ephemeris.tabulated(
        empty_ephemeris_dict,
        global_frame_origin,
        global_frame_orientation)
orbiter_ephemeris.make_multi_arc_ephemeris = True
body_settings.get("Orbiter").ephemeris_settings = orbiter_ephemeris

# Create system of bodies Enceladus
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add RatNOS spacecraft to system of bodies Enceladus
bodies.get("Orbiter").mass = 2150.0

# Create radiation pressure settings Enceladus
ref_area = 100.0
srp_coef = 1.2
occulting_bodies = {"Sun": ["Enceladus"]}
orbiter_srp_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    ref_area, srp_coef, occulting_bodies)
environment_setup.add_radiation_pressure_target_model(bodies, "Orbiter", orbiter_srp_settings)

# Define accelerations acting on Orbiter Enceladus
accelerations_settings_orbiter = dict(
    Enceladus=[
        propagation_setup.acceleration.spherical_harmonic_gravity(3, 3),
        propagation_setup.acceleration.empirical()
    ],
    Saturn=[
        propagation_setup.acceleration.spherical_harmonic_gravity(8, 8)
    ],
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ])
acceleration_settings = {"Orbiter": accelerations_settings_orbiter}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, body_to_propagate, central_body)


# Define propagation arcs during GCO (one day long) Enceladus
arc_duration = 0.5 * constants.JULIAN_DAY

arc_start_times = []
arc_end_times = []
arc_start = start_gco
while arc_start+arc_duration <= end_gco:
    arc_start_times.append(arc_start)
    arc_end_times.append(arc_start+arc_duration)
    arc_start += arc_duration

# Extract total number of (propagation) arcs during GCO
nb_arcs = len(arc_start_times)
print('Number of arcs during GCO', nb_arcs)


# Define integrator settings
time_step = 300.0
integrator_moons = propagation_setup.integrator.runge_kutta_fixed_step_size(
    time_step, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78)


# Define arc-wise initial states for Orbiter wrt. Enceladus.
# The initial states need to be provided at the start of each propagation arc.
# need to provide initial states for stable orbits in an inertial frame! Take K2 in body-fixed frame, transform to an inertial frame

# Get rotation matrix between IAU_Enceladus and global_frame_orientation
rotation_matrix = spice.compute_rotation_matrix_between_frames("IAU_Enceladus",global_frame_orientation, arc_start_times[0])
rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation,"IAU_Enceladus", arc_start_times[0] )

# Assign initial state in Cartesian coordinates in inertial frame
initial_state = np.ndarray([6])
initial_state[0:3] = [475323.709, 102991.720, -48576.955] #rotation_matrix.dot(initial_state_enceladus_fixed[0:3]) #[475323.709, 102991.720, -48576.955]
initial_state[3:6] = [3.009, 75.062, 95.705] #rotation_matrix.dot(initial_state_enceladus_fixed[3:6]) #[3.009, 75.062, 95.705]

#print("initial state cartesian inertial")
#print(initial_state)
#print("initial state cartesian fixed rotated")
#print(rotation_matrix_back.dot(initial_state[0:3]))

initial_states = []
for i in range(nb_arcs):
    initial_states.append(initial_state)
#initial_states.append(spice.get_body_cartesian_state_at_epoch("-28", "Enceladus", "J2000", "None", arc_start_times[i])) #this is Juice

# Define dependent variables to be saved during propagation
dependent_variables_names = [
    propagation_setup.dependent_variable.latitude("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.longitude("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Enceladus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Saturn"
    ),
    propagation_setup.dependent_variable.total_acceleration("Orbiter"),
    propagation_setup.dependent_variable.keplerian_state("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.altitude("Orbiter", "Enceladus")
]

# Define arc-wise propagator settings
propagator_settings_list = []
for i in range(nb_arcs):
    propagator_settings_list.append(propagation_setup.propagator.translational(
        central_body, acceleration_models, body_to_propagate, initial_states[i], arc_start_times[i], integrator_moons, propagation_setup.propagator.time_termination(arc_end_times[i]),
        propagation_setup.propagator.cowell, dependent_variables_names))
# Concatenate all arc-wise propagator settings into multi-arc propagator settings
propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

# Propagate dynamics and retrieve simulation results
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
simulation_results = simulator.propagation_results.single_arc_results


# Manually define Enceladus ground stations (s1, s2, s3)
station_names = ["s1", "s2", "s3"]

s1_longitude=180.
s1_latitude=0.
s2_longitude=190.
s2_latitude=0.
s3_longitude=185.
s3_latitude=10.

station_coordinates = {station_names[0]: [0.0, np.deg2rad(s1_longitude), np.deg2rad(s1_latitude)], station_names[1]: [0.0, np.deg2rad(s2_longitude), np.deg2rad(s2_latitude)], station_names[2]: [0.0, np.deg2rad(s3_longitude), np.deg2rad(s3_latitude)]}

#print("station coordinates", station_coordinates)

for station in station_names:
    environment_setup.add_ground_station(
        bodies.get_body("Enceladus"), station, station_coordinates[station], element_conversion.geodetic_position_type)


# Define link ends for two-way Doppler and two-way range observables, for each ground station
link_ends = []
for station in station_names:
    link_ends_per_station = dict()
    #link_ends_per_station[observation.transmitter] = observation.body_reference_point_link_end_id("Enceladus", station)
    #link_ends_per_station[observation.receiver] = observation.body_reference_point_link_end_id("Enceladus", station)
    #link_ends_per_station[observation.reflector1] = observation.body_origin_link_end_id("Orbiter")
    link_ends_per_station[observation.transmitter] = observation.body_origin_link_end_id("Orbiter")
    link_ends_per_station[observation.receiver] = observation.body_origin_link_end_id("Orbiter")
    link_ends_per_station[observation.reflector1] = observation.body_reference_point_link_end_id("Enceladus", station)
    link_ends.append(link_ends_per_station)


# Define tracking arcs (arc duration is set to 8h/day during GCO) Enceladus
# The tracking arcs are (arbitrarily) set to start 2h after the start of each propagation arc.
tracking_arc_duration = 8.0 * 3600.0
tracking_arcs_start = []
tracking_arcs_end = []
for arc_start in arc_start_times:
    tracking_arc_start = arc_start + 2.0 * 3600.0
    tracking_arcs_start.append(tracking_arc_start)
    tracking_arcs_end.append(tracking_arc_start + tracking_arc_duration)


# Create observation settings for each link ends and observable

# Define light-time calculations settings
light_time_correction_settings = observation.first_order_relativistic_light_time_correction(["Sun"])

# Define range biases settings
biases = []
for i in range(nb_arcs):
    biases.append(np.array([0.0]))
range_bias_settings = observation.arcwise_absolute_bias(tracking_arcs_start, biases, observation.receiver)

# Define observation settings list
observation_settings_list = []
for link_end in link_ends:
    link_definition = observation.LinkDefinition(link_end)
    observation_settings_list.append(observation.n_way_doppler_averaged(link_definition, [light_time_correction_settings]))
    observation_settings_list.append(observation.two_way_range(link_definition, [light_time_correction_settings], range_bias_settings))


# Define observation simulation times for both Doppler and range observables
doppler_cadence = 60.0
range_cadence = 300.0

observation_times_doppler = []
observation_times_range = []
for i in range(nb_arcs):
    # Doppler observables
    time = tracking_arcs_start[i]
    while time <= tracking_arcs_end[i]:
        observation_times_doppler.append(time)
        time += doppler_cadence
    # Range observables
    time = tracking_arcs_start[i]
    while time <= tracking_arcs_end[i]:
        observation_times_range.append(time)
        time += range_cadence

observation_times_per_type = dict()
observation_times_per_type[observation.n_way_averaged_doppler_type] = observation_times_doppler
observation_times_per_type[observation.n_way_range_type] = observation_times_range


# Define observation settings for both observables, and all link ends (i.e., all ground stations)
observation_simulation_settings = []
for link_end in link_ends:
    # Doppler
    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.n_way_averaged_doppler_type, observation.LinkDefinition(link_end), observation_times_per_type[observation.n_way_averaged_doppler_type]))
    # Range
    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.n_way_range_type, observation.LinkDefinition(link_end), observation_times_per_type[observation.n_way_range_type]))


# Create viability settings which define when an observation is feasible
viability_settings = []

# For all tracking stations, check if elevation is sufficient Enceladus
for station in station_names:
    viability_settings.append(observation.elevation_angle_viability(["Enceladus", station], np.deg2rad(10.0)))
# Check whether Enceladus or Saturn are occulting the signal
viability_settings.append(observation.body_occultation_viability(["Orbiter", ""], "Enceladus"))
#viability_settings.append(observation.body_occultation_viability(["Orbiter", ""], "Saturn"))
# Check whether SEP angle is sufficiently large
viability_settings.append(observation.body_avoidance_viability(["Orbiter", ""], "Sun", np.deg2rad(5.0)))

# Apply viability checks to all simulated observations
observation.add_viability_check_to_all(observation_simulation_settings, viability_settings)


## Define parameters to estimate

## Add arc-wise initial states of the Orbiter spacecraft wrt Enceladus
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, arc_start_times)

## Add Enceladus's gravitational parameter
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Enceladus"))

## Add Enceladus's gravity field spherical harmonics coefficients
max_deg_enceladus_gravity = 3
parameter_settings.append(estimation_setup.parameter.spherical_harmonics_c_coefficients("Enceladus", 2, 0, max_deg_enceladus_gravity, max_deg_enceladus_gravity))
parameter_settings.append(estimation_setup.parameter.spherical_harmonics_s_coefficients("Enceladus", 2, 1, max_deg_enceladus_gravity, max_deg_enceladus_gravity))

## Add Enceladus's rotational parameters # Don't activate this
## parameter_settings.append(estimation_setup.parameter.constant_rotation_rate("Enceladus"))
## parameter_settings.append(estimation_setup.parameter.rotation_pole_position("Enceladus"))

## Add arc-wise empirical accelerations acting on the Orbiter spacecraft Enceladus
acc_components = {estimation_setup.parameter.radial_empirical_acceleration_component: [estimation_setup.parameter.constant_empirical],
                  estimation_setup.parameter.along_track_empirical_acceleration_component: [estimation_setup.parameter.constant_empirical],
                  estimation_setup.parameter.across_track_empirical_acceleration_component: [estimation_setup.parameter.constant_empirical]}
parameter_settings.append(estimation_setup.parameter.arcwise_empirical_accelerations("Orbiter", "Enceladus", acc_components, arc_start_times))

## Add ground stations' positions
for station in station_names:
    parameter_settings.append(estimation_setup.parameter.ground_station_position("Enceladus", station))


## # Add arc-wise range biases as consider parameters # Don't activate this
## for link_end in link_ends:
##     parameter_settings.append(estimation_setup.parameter.arcwise_absolute_observation_bias(
##         observation.LinkDefinition(link_end), observation.n_way_range_type, tracking_arcs_start, observation.receiver))


## # Define consider parameters (COMMENTED FOR NOW BECAUSE OF OBS BIAS PARTIAL ISSUE) # Don't activate this
## # Add arc-wise range biases as consider parameters
## consider_parameters_settings = []
## for link_end in link_ends:
##     consider_parameters_settings.append(estimation_setup.parameter.arcwise_absolute_observation_bias(
##         observation.LinkDefinition(link_end), observation.n_way_range_type, tracking_arcs_start, observation.receiver))


# Create parameters to estimate object Enceladus
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings) #, consider_parameters_settings)
estimation_setup.print_parameter_names(parameters_to_estimate)
nb_parameters = len(parameters_to_estimate.parameter_vector)
print("Number of parameters to estimate", len(parameters_to_estimate.parameter_vector))
print(parameters_to_estimate.parameter_vector[1813:1816])
print(parameters_to_estimate.parameter_vector[1816:1819])
print(parameters_to_estimate.parameter_vector[1819:1822])

# Create the estimator
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings_list, propagator_settings)

# Simulate all observations
simulated_observations = estimation.simulate_observations(observation_simulation_settings, estimator.observation_simulators, bodies)


# Define a priori covariance matrix # values are uncertainties
inv_apriori = np.zeros((nb_parameters, nb_parameters))

# Set a priori constraints for Orbiter state(s)
a_priori_position = 5.0e3
a_priori_velocity = 0.5
indices_states = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.arc_wise_initial_body_state_type, ("Orbiter", "")))[0]
for i in range(indices_states[1]//6):
    for j in range(3):
        inv_apriori[indices_states[0]+i*6+j, indices_states[0]+i*6+j] = a_priori_position**-2  # a priori position
        inv_apriori[indices_states[0]+i*6+j+3, indices_states[0]+i*6+j+3] = a_priori_velocity**-2  # a priori velocity

# Set a priori constraint for Enceladus's gravitational parameter
a_priori_mu = 0.0211292085479989E+9 #7.211292085479989E+9 #0.03e9 #this is the uncertainty on gm see Cassini results for Enceladus
indices_mu = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.gravitational_parameter_type, ("Enceladus", "")))[0]
for i in range(indices_mu[1]):
    inv_apriori[indices_mu[0]+i, indices_mu[0]+i] = a_priori_mu**-2

## Set a priori constraint for Enceladus's gravity field coefficients
## ERROR IN EXPOSE FUNCTION (RETURN SINE COEFFICIENTS INDICES), TEMPORARY FIX FOR NOW
nb_cosine_coef = (max_deg_enceladus_gravity+1) * (max_deg_enceladus_gravity+2) // 2 - 3  # the minus 3 accounts for degrees 0 and 1 coefficients which are not estimated
indices_cosine_coef = (nb_arcs*6+1, nb_cosine_coef)
## indices_cosine_coef = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.spherical_harmonics_cosine_coefficient_block_type, ("Enceladus", "")))[0]
indices_sine_coef = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.spherical_harmonics_sine_coefficient_block_type, ("Enceladus", "")))[0]

# Apply Kaula's constraint to Enceladus's gravity field a priori
kaula_constraint_multiplier = 4.0e-4
apply_kaula_constraint_a_priori(kaula_constraint_multiplier, max_deg_enceladus_gravity, indices_cosine_coef, indices_sine_coef, inv_apriori)

## Overwrite Kaula's rule with existing uncertainties for C20 and C22
apriori_C20 = 1.e-6 #2.9e-6
apriori_C22 = 1.e-7 #0.87e-6
inv_apriori[indices_cosine_coef[0], indices_cosine_coef[0]] = apriori_C20**-2
inv_apriori[indices_cosine_coef[0]+2, indices_cosine_coef[0]+2] = apriori_C22**-2

## Set tight constraint for C21, S21, and S22
inv_apriori[indices_cosine_coef[0]+1, indices_cosine_coef[0]+1] = 1.0e-12**-2
inv_apriori[indices_sine_coef[0], indices_sine_coef[0]] = 1.0e-12**-2
inv_apriori[indices_sine_coef[0]+1, indices_sine_coef[0]+1] = 1.0e-12**-2

# Set a priori constraints for Enceladus's rotational rate # Don't activate this
## a_priori_rotation_rate =
## indices_rotation_rate = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.constant_rotation_rate_type, ("Enceladus", "")))[0]
## for i in range(indices_rotation_rate[0][1]):
##     inv_apriori[indices_rotation_rate[0][0]+i, indices_rotation_rate[0][0]+i] = 1.0/a_priori_rotation_rate**2

# # A priori rotation pole of Enceladus # Don't activate this
## a_priori_pole_ra = 4.0e-5
## a_priori_pole_dec = 5.0e-5
## indices_rotation_pole = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.rotation_pole_position_type, ("Enceladus", "")))[0]
## for i in range(indices_rotation_pole[1]//2):
##     inv_apriori[indices_rotation_pole[0]+i*2, indices_rotation_pole[0]+i*2] = a_priori_pole_ra**-2  # a priori RA
##     inv_apriori[indices_rotation_pole[0]+i*2+1, indices_rotation_pole[0]+i*2+1] = a_priori_pole_dec **-2  # a priori DEC



# Set a priori constraints for empirical accelerations acting on Orbiter
a_priori_emp_acc = 1.0e-7
indices_emp_acc = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.arc_wise_empirical_acceleration_coefficients_type, ("Orbiter", "Enceladus")))[0]
for i in range(indices_emp_acc[1]):
    inv_apriori[indices_emp_acc[0] + i, indices_emp_acc[0] + i] = a_priori_emp_acc ** -2

# Set a priori constraints for ground station positions
a_priori_station = 0.03 # this is the last uncertainty
for station in station_names:
    indices_station_pos = parameters_to_estimate.indices_for_parameter_type((estimation_setup.parameter.ground_station_position_type, ("Enceladus", station)))[0]
    for i in range(indices_station_pos[1]):
        inv_apriori[indices_station_pos[0] + i, indices_station_pos[0] + i] = a_priori_station ** -2

# Retrieve full vector of a priori constraints
apriori_constraints = np.reciprocal(np.sqrt(np.diagonal(inv_apriori)))
## print('A priori constraints')
## print(apriori_constraints)

## COMMENTED FOR NOW BECAUSE OF OBS BIAS PARTIAL ISSUE # Don't activate this
## # Define consider parameters covariance
## nb_consider_parameters = nb_arcs*len(station_names)
## consider_parameters_covariance = np.zeros((nb_consider_parameters, nb_consider_parameters))
##
## # Set consider covariance for range biases for all three ESTRACT stations # Don't activate this
## a_priori_biases = 2.0
## for station in station_names:
##     indices_biases = (0, nb_arcs*len(station_names))
##     for i in range(indices_biases[1]):
##         consider_parameters_covariance[indices_biases[0] + i, indices_biases[0] + i] = a_priori_biases ** 2


# Define covariance input settings
covariance_input = estimation.CovarianceAnalysisInput(simulated_observations, inv_apriori) #, consider_parameters_covariance)
covariance_input.define_covariance_settings(reintegrate_variational_equations=False, save_design_matrix=True)

# Apply weights to simulated observations
doppler_noise = 12.0e-6
range_noise = 0.2
weights_per_observable = {observation.n_way_averaged_doppler_type: doppler_noise ** -2,
                          observation.n_way_range_type: range_noise ** -2}
covariance_input.set_constant_weight_per_observable(weights_per_observable)

# Perform the covariance analysis
covariance_output = estimator.compute_covariance(covariance_input)

# Retrieve covariance results
correlations = covariance_output.correlations
covariance = covariance_output.covariance
formal_errors = covariance_output.formal_errors
partials = covariance_output.weighted_design_matrix

# Print the formal errors
#print('Formal errors')
#print(covariance_output.formal_errors)

## # COMMENTED FOR NOW NO CONSIDER PARAMETERS BECAUSE OF OBS BIAS PARTIAL ISSUE # Don't activate this
## # Retrieve results with consider parameters
## consider_covariance_contribution = covariance_output.consider_covariance_contribution
## covariance_with_consider_parameters = covariance_output.unnormalized_covariance_with_consider_parameters
## formal_errors_with_consider_parameters = np.sqrt(np.diagonal(covariance_with_consider_parameters))
## # Compute correlations with consider parameters
## correlations_with_consider_parameters = covariance_with_consider_parameters
## for i in range(nb_parameters):
##     for j in range(nb_parameters):
##         correlations_with_consider_parameters[i, j] /= (formal_errors_with_consider_parameters[i] * formal_errors_with_consider_parameters[j])

## # Propagate formal errors (TO BE FIXED, NOT YET POSSIBLE FOR MULTI-ARC) # Don't activate this
## output_times = np.arange(start_gco, end_gco, 3600.0)
## propagated_formal_errors = estimation.propagate_formal_errors_rsw_split_output(covariance_output, estimator, output_times)

## PLOTS

# Get simulation results over first propagation arc
simulation_results_first_arc = simulation_results[0]
propagated_state_first_arc = result2array(simulation_results_first_arc.state_history)
dependent_variables_first_arc = result2array(simulation_results_first_arc.dependent_variable_history)

### Conversion of Cartesian elements to the body-fixed frame for each time step
print("format of propagated_state_first_arc", np.shape(propagated_state_first_arc))
num_rows = np.shape(propagated_state_first_arc)[0]
num_columns = np.shape(propagated_state_first_arc)[1]

propagated_state_first_arc_body_fixed = np.ndarray([num_rows,7])

for i in range(num_rows):
    rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation, "IAU_Enceladus", propagated_state_first_arc[i,0])
    propagated_state_first_arc_body_fixed[i,0] = propagated_state_first_arc[i,0]
    propagated_state_first_arc_body_fixed[i,1:4] = rotation_matrix_back.dot(propagated_state_first_arc[i,1:4])
    propagated_state_first_arc_body_fixed[i,4:7] = rotation_matrix_back.dot(propagated_state_first_arc[i,4:7])

fig = plt.figure(figsize=(6,6), dpi=400)
# Plot trajectory of Orbiter during first propagation arc
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('ortho')
ax.set_aspect('equal')
ax.plot(propagated_state_first_arc_body_fixed[:, 1]/1e3, propagated_state_first_arc_body_fixed[:, 2]/1e3, propagated_state_first_arc_body_fixed[:, 3]/1e3, linestyle='-', color='black', linewidth=0.5)
ax.set_title('Orbiter orbit wrt Enceladus over one period')
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_zlim(-500, 500)
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_xticks([])
ax.set_yticks([-400, -200, 0, 200, 400])
ax.set_zticks([-400, -200, 0, 200, 400])
#ax.grid(False)
ax.view_init(0, 0)
ax.grid()

# Plot Orbiter ground track during first propagation arc
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
enceladus_map = '../propagation/enceladus_map03602.jpg'
ax.imshow(plt.imread(enceladus_map), extent = [0, 360, -90, 90])

# Resolve 2pi ambiguity longitude
for k in range(len(dependent_variables_first_arc)):
    if dependent_variables_first_arc[k, 2] < 0:
        dependent_variables_first_arc[k, 2] = dependent_variables_first_arc[k, 2] + 2.0 * np.pi
ax.plot(dependent_variables_first_arc[:, 2]*180/np.pi, dependent_variables_first_arc[:, 1]*180.0/np.pi, '.', markersize=1.0, color='blue', fillstyle='full')
plt.scatter(s1_longitude, s1_latitude)
plt.scatter(s2_longitude, s2_latitude)
plt.scatter(s3_longitude, s3_latitude)
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.set_xticks(np.arange(0, 361, 40))
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_title('Orbiter ground track over one day')


# Plot weighted partials
plt.figure(figsize=(9, 6), dpi=400)
plt.imshow(np.log10(np.abs(partials)), aspect='auto', interpolation='none')
cb = plt.colorbar()
cb.set_label('log10(weighted partials)')
plt.title("Weighted partials")
plt.ylabel("Index - Observation")
plt.xlabel("Index - Estimated Parameter")
plt.tight_layout()

# # Plot contribution of consider parameters to formal errors
# plt.figure()
# plt.plot(apriori_constraints, label='apriori constraints')
# plt.plot(formal_errors, label='nominal errors')
# plt.plot(formal_errors_with_consider_parameters, label='errors w/ consider parameters')
# plt.grid()
# plt.yscale("log")
# plt.ylabel('Formal errors')
# plt.xlabel('Index parameter [-]')
# plt.legend()
# plt.title('Effect consider parameters')

# Plot correlations (default)
plt.figure(figsize=(9, 6), dpi=400)
plt.imshow(np.abs(correlations), aspect='auto', interpolation='none')
plt.colorbar()
plt.title("Correlations")
plt.xlabel("Index - Estimated Parameter")
plt.ylabel("Index - Estimated Parameter")
plt.tight_layout()

# # Plot correlations (incl. contribution of consider parameters)
# plt.figure(figsize=(9, 6))
# plt.imshow(np.abs(correlations_with_consider_parameters), aspect='auto', interpolation='none')
# plt.colorbar()
# plt.title("Correlations w/ consider parameters")
# plt.xlabel("Index - Estimated Parameter")
# plt.ylabel("Index - Estimated Parameter")
# plt.tight_layout()


# Retrieve Doppler observation times for the first arc
sorted_observations = simulated_observations.sorted_observation_sets
# doppler_obs_times_new_forcia_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times if t <= start_gco+arc_duration]
# doppler_obs_times_cebreros_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times if t <= start_gco+arc_duration]
doppler_obs_times_s1_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times if t <= start_gco+arc_duration]
doppler_obs_times_s2_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times if t <= start_gco+arc_duration]
doppler_obs_times_s3_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][2][0].observation_times if t <= start_gco+arc_duration]

print("sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times", np.shape(sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times))
print("sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times", np.shape(sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times))
print("sorted_observations[observation.n_way_averaged_doppler_type][2][0].observation_times", np.shape(sorted_observations[observation.n_way_averaged_doppler_type][2][0].observation_times))

# Plot observation times (for now only for s3, but designed to eventually include all three stations)
plt.figure(dpi=400)
# plt.plot(doppler_obs_times_new_forcia_first_arc, np.ones((len(doppler_obs_times_new_forcia_first_arc),1 )))
# plt.plot(doppler_obs_times_cebreros_first_arc, 2.0 * np.ones((len(doppler_obs_times_cebreros_first_arc),1 )))
plt.plot(doppler_obs_times_s1_first_arc, 1.0 * np.ones((len(doppler_obs_times_s1_first_arc),1 )))
plt.plot(doppler_obs_times_s2_first_arc, 2.0 * np.ones((len(doppler_obs_times_s2_first_arc),1 )))
plt.plot(doppler_obs_times_s3_first_arc, 3.0 * np.ones((len(doppler_obs_times_s3_first_arc),1 )))
plt.xlabel('Observation times [h]')
plt.ylabel('')
plt.yticks([1, 2, 3], ['S1', 'S2', 'S3'])
plt.ylim([0.5, 3.5])
plt.title('Viable observations over first arc')
plt.grid()


# Plot gravity field spectrum (a priori + formal errors)
# Extract a priori cosine and sine coefs
apriori_cosine_coefs = np.reciprocal(np.sqrt(inv_apriori.diagonal()))[indices_cosine_coef[0]:indices_cosine_coef[0]+indices_cosine_coef[1]]
apriori_sine_coefs = np.reciprocal(np.sqrt(inv_apriori.diagonal()))[indices_sine_coef[0]:indices_sine_coef[0]+indices_sine_coef[1]]

# Extract formal errors for Enceladus's cosine and sine coefficients
formal_errors_cosine_coefs = covariance_output.formal_errors[indices_cosine_coef[0]:indices_cosine_coef[0]+indices_cosine_coef[1]]
formal_errors_sine_coefs = covariance_output.formal_errors[indices_sine_coef[0]:indices_sine_coef[0]+indices_sine_coef[1]]

# Initialise empty vectors for gravity field coefficients' a priori constraints and errors
apriori_cosine_per_deg = np.zeros(max_deg_enceladus_gravity-1)
apriori_sine_per_deg = np.zeros(max_deg_enceladus_gravity-1)
formal_errors_cosine_per_deg = np.zeros(max_deg_enceladus_gravity-1)
formal_errors_sine_per_deg = np.zeros(max_deg_enceladus_gravity-1)

# Compute rms of a priori constraints and formal errors for gravity field coefficients
start_index_cosine_deg = 0
start_index_sine_deg = 0
for deg in range(2, max_deg_enceladus_gravity+1):

    # Cosine coefficients
    rms_apriori = 0
    rms_error = 0
    for j in range(deg+1):
        rms_apriori += getKaulaConstraint(kaula_constraint_multiplier, deg)**2
        rms_error += formal_errors_cosine_coefs[start_index_cosine_deg+j]**2
    start_index_cosine_deg += deg+1

    apriori_cosine_per_deg[deg-2] = np.sqrt(rms_apriori/(deg+1))
    formal_errors_cosine_per_deg[deg-2] = np.sqrt(rms_error/(deg+1))

    # Sine coefficients
    rms_apriori = 0
    rms_error = 0
    for j in range(deg):
        rms_apriori += getKaulaConstraint(kaula_constraint_multiplier, deg)
        rms_error += formal_errors_sine_coefs[start_index_sine_deg+j]**2
    start_index_sine_deg += deg

    apriori_sine_per_deg[deg-2] = np.sqrt(rms_apriori/deg)
    formal_errors_sine_per_deg[deg-2] = np.sqrt(rms_error/deg)


# Plot Enceladus's gravity spectrum
plt.figure(dpi=400)
plt.plot(apriori_cosine_per_deg, label='Cosine Kaula constraint')
plt.plot(formal_errors_cosine_per_deg, label='Cosine estimated errors')
plt.plot(apriori_sine_per_deg, label='Sine Kaula constraint')
plt.plot(formal_errors_sine_per_deg, label='Sine estimated errors')
plt.grid()
plt.yscale("log")
plt.ylabel('RMS gravity coefficients')
plt.xlabel('Degree')
plt.xticks(np.arange(0, max_deg_enceladus_gravity-1, 1), np.arange(2, max_deg_enceladus_gravity+1, 1))
plt.title('Gravity spectrum Enceladus')
plt.legend()

print("apriori_cosine_per_deg", apriori_cosine_per_deg)
print("apriori_sine_per_deg", apriori_sine_per_deg)

#print("formal_errors_cosine_per_deg", formal_errors_cosine_per_deg)
#print("formal_errors_sine_per_deg", formal_errors_sine_per_deg)

# Show all plots
plt.show()

### Accelerations over time
"""
plt.figure(figsize=(9, 5))

# Spherical Harmonics Gravity Acceleration Enceladus
time_hours = dependent_variables_first_arc[:,0]/3600
acceleration_norm_sh_enceladus = dependent_variables_first_arc[:, 3]
plt.plot(time_hours, acceleration_norm_sh_enceladus, label='SH Enceladus')

# Spherical Harmonics Gravity Acceleration Saturn
acceleration_norm_sh_saturn = dependent_variables_first_arc[:, 4]
plt.plot(time_hours, acceleration_norm_sh_saturn, label='SH Saturn')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend(bbox_to_anchor=(1.005, 1))
plt.suptitle("Accelerations norms on Orbiter, distinguished by type and origin, over the course of propagation.")
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()
"""
print(parameters_to_estimate.parameter_vector[1813:1816])
print(parameters_to_estimate.parameter_vector[1816:1819])
print(parameters_to_estimate.parameter_vector[1819:1822])
