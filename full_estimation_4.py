# DELFI-C3 - Parameter Estimation
## Context
"""
Within this example we will focus on how to set up and perform the full estimation of a spacecraft's initial state, drag
coefficient, and radiation pressure coefficient. Using radars (stations), we will simulate a tracking routine of the
spacecraft using a series of open-loop Doppler range-rate measurements at x mm/s every xx sec. To assure an uninterrupted
line-of-sight bw the radar and the spacecraft, a min  elevation angle of >xx° above the horizon (as seen from the station)
will be imposed as constraint on the simulation of observations.
"""

## Import statements

# Load required standard modules
import numpy as np
import math
import pyproj
import itertools
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.kernel.astro import gravitation
from tudatpy import astro

## Define a set of radar positions for grid search

nposit = 3 # number of positions
nradars = 2 # number of radars per iteration

# Function for converting cartesian coordinates to geodetic ones
def cartesian_to_geodetic(x, y, z): # not used
    enc_rad = 256600.0
    alt = enc_rad
    # Define the coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # Perform the coordinate transformation
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return lat, lon, alt

# Function for defining equidistant radar positions set along a Fibonacci spiral
def fibonacci_sphere(samples=10):
    pointsgeo = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2 # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        longi = np.rad2deg(math.atan2(y, x)) #* 360. / 2 * math.pi
        latit = np.rad2deg(math.asin(z / 1.)) #* 360 / 2 * math.pi
        #points.append((x, y, z))
        pointsgeo.append(longi)
        pointsgeo.append(latit)

    return pointsgeo

# Geodetic coordinates of the radar positions set
longi=fibonacci_sphere(nposit)[0::2]
latit=fibonacci_sphere(nposit)[1::2]

radars=[]
for i in range(len(longi)):
    radars.append((longi[i],latit[i]))

print(len(longi))
print(radars)

radars_combis = itertools.combinations(radars, nradars)

## Convert the iterator to a list to display all combinations
combis_list = list(radars_combis)

## Print the result
print("All possible combinations of 2 radars:")
print(len(combis_list))
print(combis_list)

"""
First, NAIF's `SPICE` kernels are loaded, to make the positions of various bodies such as the Enceladus, the Sun, 
and Saturn known to `tudatpy`.
Subsequently, the start and end epoch of the simulation are defined. Note that using `tudatpy`, the times are generally 
specified in seconds since J2000. Hence, setting the start epoch to `0` corresponds to the 1st of January 2000. 
The end epoch specifies a total duration of the simulation.
For more information on J2000 and the conversion between different temporal reference frames, please refer to the 
API documentation of the [`time_conversion module`](https://tudatpy.readthedocs.io/en/latest/time_conversion.html).
"""

# Load spice kernels
spice.load_standard_kernels()
kernels = ['/home/neumwl/TudatProjects/de438.bsp', '/home/neumwl/TudatProjects/sat427.bsp',
           '/home/neumwl/TudatProjects/pck00010.tpc']
spice.load_standard_kernels(kernels)

#Ab hier eine for-Schleife für die Definition der stations und die estimation procedure

normdiff=[]

## Configuration

## Set up the environment
"""
We will now create and define the settings for the environment of our simulation. In particular, this covers 
the creation of (celestial) bodies, vehicle(s), and environment interfaces.
"""

### Create the main bodies
"""
To create the systems of bodies for the simulation, one first has to define a list of strings of all bodies 
that are to be included. Note that the default body settings (such as atmosphere, body shape, rotation model) 
are taken from the `SPICE` kernel.
These settings, however, can be adjusted. Please refer to the [Available Environment Models]
(https://tudat-space.readthedocs.io/en/latest/_src_user_guide/state_propagation/environment_setup/create_models/available.html#available-environment-models) 
in the user guide for more details.
Finally, the system of bodies is created using the settings. This system of bodies is stored into the variable `bodies`.
"""

# Set simulation start and end epochs
start_gco = 0. * constants.JULIAN_YEAR  # beginning circular orbital phase 35.3844 * constants.JULIAN_YEAR
# noinspection PyUnboundLocalVariable
end_gco = start_gco + 1.35 * constants.JULIAN_DAY #13.5 * constants.JULIAN_DAY # 35.73 * constants.JULIAN_YEAR  # end circular orbital phase

# Create default body settings for bodies_to_create, with "Enceladus"/"J2000" as the global frame origin and orientation
global_frame_origin = "Enceladus"
global_frame_orientation = "J2000"
bodies_to_propagate = ["Orbiter"]
central_bodies = ["Enceladus"]

# Create default body settings for "Sun", "Enceladus", and "Saturn"
bodies_to_create = ["Sun", "Enceladus", "Saturn"]
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

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
body_settings.get("Saturn").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    saturn_gravitational_parameter,
    saturn_reference_radius,
    saturn_normalized_cosine_coefficients,
    saturn_normalized_sine_coefficients,
    saturn_associated_reference_frame )

# Add setting for moment of inertia for Saturn
body_settings.get("Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210

# Compute rotation rate for Enceladus
mu_saturn = saturn_gravitational_parameter #spice.get_body_properties("Saturn", "GM", 1)[0] * 1.0e9
initial_state_enceladus = spice.get_body_cartesian_state_at_epoch("Enceladus", "Saturn", "J2000", "None", start_gco)
keplerian_state_enceladus = element_conversion.cartesian_to_keplerian(initial_state_enceladus, mu_saturn)
rotation_rate_enceladus = np.sqrt(mu_saturn/keplerian_state_enceladus[0]**3)

# Set rotation model settings Enceladus
initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000", "IAU_Enceladus", start_gco)
body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
    "J2000", "IAU_Enceladus", initial_orientation_enceladus, start_gco, rotation_rate_enceladus)

# Gravity field settings
body_settings.get("Enceladus").gravity_field_settings = get_gravity_enceladus()

#bodies.create_empty_body("Orbiter")
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

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

### Create the vehicle and its environment interface
"""
We will now create the satellite - called Orbiter - for which an orbit will be simulated. Using an `empty_body` 
as a blank canvas for the satellite, we define mass of 2150 kg, a reference area (used both for aerodynamic 
and radiation pressure) of 100 m$^2$, and a aerodynamic drag coefficient of 1.2. Idem for the radiation 
pressure coefficient. Finally, when setting up the radiation pressure interface, Enceladus is set as a body 
that can occult the radiation emitted by the Sun.
"""

# Create vehicle objects.
bodies.get("Orbiter").mass = 2150.0

# Create aerodynamic coefficient interface settings
reference_area = 0.0#100.0 #(4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0.0, 0.0]
)
# Add the aerodynamic interface to the environment
environment_setup.add_aerodynamic_coefficient_interface(bodies, "Orbiter", aero_coefficient_settings)

# Create radiation pressure settings
reference_area = 100.0 #(4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies = ["Enceladus"] #occulting_bodies = {"Sun": ["Enceladus"]}
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
)
# Add the radiation pressure interface to the environment
environment_setup.add_radiation_pressure_interface(bodies, "Orbiter", radiation_pressure_settings)

## Set up the propagation
"""
Having the environment created, we will define the settings for the propagation of the spacecraft. 
First, we have to define the body to be propagated - here, the spacecraft - and the central body - here, 
Enceladus - with respect to which the state of the propagated body is defined.
"""

### Create the acceleration model
"""
Subsequently, all accelerations (and there settings) that act on `Orbiter` have to be defined. In particular, we will consider:
* Gravitational acceleration using a spherical harmonic approximation for Enceladus and Saturn.
* Empirical acceleration for Enceladus.
* Gravitational acceleration using a simple point mass model for the Sun.
* Radiation pressure experienced by the spacecraft - shape-wise approximated as a spherical cannonball - due to the Sun.

The defined acceleration settings are then applied to `Orbiter` by means of a dictionary, which is finally used 
as input to the propagation setup to create the acceleration models.
"""

# Define the accelerations acting on Orbiter
accelerations_settings_orbiter = dict(
    Sun=[
        propagation_setup.acceleration.cannonball_radiation_pressure(),
        # propagation_setup.acceleration.radiation_pressure()
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Saturn=[
        propagation_setup.acceleration.spherical_harmonic_gravity(8, 8)
    ],
    Enceladus=[
        propagation_setup.acceleration.spherical_harmonic_gravity(3, 3),
        propagation_setup.acceleration.empirical()
    ])

# Create global accelerations dictionary
acceleration_settings = {"Orbiter": accelerations_settings_orbiter}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

# Define propagation arcs during GCO (one day long) Enceladus
arc_duration = 0.8*(end_gco-start_gco) #0.5 * constants.JULIAN_DAY
print('arc duration', arc_duration/3600.0)

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

### Define the initial states for Orbiter wrt. Enceladus.
"""
Realise that the initial state of the spacecraft always has to be provided as a cartesian state - i.e. in the form 
of a list with the first three elements representing the initial position, and the three remaining elements 
representing the initial velocity.
"""
# The initial states need to be provided at the start of each propagation arc. We need to provide the initial states
# for stable orbits in an inertial frame! Take K2 in body-fixed frame, transform to an inertial frame.

# Get rotation matrix between IAU_Enceladus and global_frame_orientation
rotation_matrix = spice.compute_rotation_matrix_between_frames("IAU_Enceladus",global_frame_orientation, arc_start_times[0])
rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation,"IAU_Enceladus", arc_start_times[0] )

# Assign initial state in Cartesian coordinates in inertial frame
initial_state = np.ndarray([6])
#initial_state[0:3] = [475323.709, 102991.720, -48576.955]
#initial_state[3:6] = [3.009, 75.062, 95.705]

#initial_state = [4.37860747e+05, 9.48716723e+04, -4.47516654e+04, 6.67354927e-01, 7.94830037e+01,  9.86040217e+01] # k1'
initial_state = [4.71789124e+05, 1.02222980e+05, -4.82193361e+04, 3.14085658e+00,  7.66580975e+01,  9.50418662e+01] # k2'
#initial_state = [4.60709175e+05,  9.98223410e+04, -4.70869558e+04,  2.46463058e+00, 7.56765950e+01,  9.76068794e+01] # k3'

#print("initial state cartesian inertial")
#print(initial_state)
#print("initial state cartesian fixed rotated")
#print(rotation_matrix_back.dot(initial_state[0:3]))

initial_states = []
for i in range(nb_arcs):
    initial_states.append(initial_state)

### Create the integrator settings
"""
For the problem at hand, we will use an RKF78 integrator with a fixed step-size of 60 seconds. 
This can be achieved by tweaking the implemented RKF78 integrator with variable step-size such that both 
the minimum and maximum step-size is equal to 60 seconds and a tolerance of 1.0
"""
# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.\
    runge_kutta_fixed_step_size(initial_time_step=10.0,
                                coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

# Define dependent variables to be saved during propagation
"""dependent_variables_names = [
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
]"""

### Create the propagator settings
"""
By combining all of the above-defined settings we can define the settings for the propagator to simulate 
the orbit of `Orbiter` around Enceladus. A termination condition needs to be defined so that the propagation stops 
as soon as the specified end epoch is reached. Finally, the translational propagator's settings are created.
"""
# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(end_gco)

# Create propagation settings
# Define arc-wise propagator settings
propagator_settings_list = []
for i in range(nb_arcs):
    propagator_settings_list.append(propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate, initial_states[i], arc_start_times[i],
        integrator_settings, propagation_setup.propagator.time_termination(arc_end_times[i])))#,
#        propagation_setup.propagator.cowell, dependent_variables_names)


# Concatenate all arc-wise propagator settings into multi-arc propagator settings
propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

# This is from ratnos_orbital_phase: Propagate dynamics and retrieve simulation results
#simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
#simulation_results = simulator.propagation_results.single_arc_results

## Set up the observations
"""
Having set the underlying dynamical model of the simulated orbit, we can define the observational model. 
Generally, this entails the addition all required ground stations, the definition of the observation links and types, 
as well as the precise simulation settings.
"""

### Add a ground station
"""
Trivially, the simulation of observations requires the extension of the current environment by at least one observer - 
a ground station. For this example, we will model three ground stations on the surface of Enceladus.
More information on how to use the `add_ground_station()` function can be found in the respective [API documentation]
(https://tudatpy.readthedocs.io/en/latest/environment_setup.html#tudatpy.numerical_simulation.environment_setup.add_ground_station).
"""

# Manually define the names and positions of Enceladus ground stations (s1, s2, s3)
station_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]

s1_altitude=0.
s1_longitude=-10.
s1_latitude=60.
s2_altitude=0.
s2_longitude=50.
s2_latitude=0.
s3_altitude=0.
s3_longitude=92.
s3_latitude=-60.
s4_altitude=0.
s4_longitude=140.
s4_latitude=0.
s5_altitude=0.
s5_longitude=180.
s5_latitude=40.
s6_altitude=0.
s6_longitude=-140.
s6_latitude=60.
s7_altitude=0.
s7_longitude=-100.
s7_latitude=50.
s8_altitude=0.
s8_longitude=-60.
s8_latitude=0.
s9_altitude=0.
s9_longitude=-20.
s9_latitude=-50.
s10_altitude=0.
s10_longitude=0.
s10_latitude=-90.

station_coordinates = {station_names[0]: [0.0, np.deg2rad(s1_latitude), np.deg2rad(s1_longitude)],
                       station_names[1]: [0.0, np.deg2rad(s2_latitude), np.deg2rad(s2_longitude)],
                       station_names[2]: [0.0, np.deg2rad(s3_latitude), np.deg2rad(s3_longitude)],
                       station_names[3]: [0.0, np.deg2rad(s4_latitude), np.deg2rad(s4_longitude)],
                       station_names[4]: [0.0, np.deg2rad(s5_latitude), np.deg2rad(s5_longitude)],
                       station_names[5]: [0.0, np.deg2rad(s6_latitude), np.deg2rad(s6_longitude)],
                       station_names[6]: [0.0, np.deg2rad(s7_latitude), np.deg2rad(s7_longitude)],
                       station_names[7]: [0.0, np.deg2rad(s8_latitude), np.deg2rad(s8_longitude)],
                       station_names[8]: [0.0, np.deg2rad(s9_latitude), np.deg2rad(s9_longitude)],
                       station_names[9]: [0.0, np.deg2rad(s10_latitude), np.deg2rad(s10_longitude)]}#"""

print("station coordinates", station_coordinates)

# Add the ground stations to the environment
for station in station_names:
    environment_setup.add_ground_station(
        bodies.get_body("Enceladus"), station, station_coordinates[station], element_conversion.geodetic_position_type)

### Define Observation Links and Types
"""
To establish the links between our ground station and `Orbiter`, we will make use of the [observation module]
(https://py.api.tudat.space/en/latest/observation.html#observation) of tudat. During th link definition, 
each member is assigned a certain function within the link, for instance as "transmitter", "receiver", or "reflector". 
Once two (or more) members are connected to a link, they can be used to simulate observations along this particular link. 
The precise type of observation made along this link - e.g., range, range-rate, angular position, etc. 
- is then determined by the chosen observable type.

To fully define an observation model for a given link, we have to create a list of the observation model settings of all 
desired observable types and their associated links. This list will later be used as input to the actual estimator object.

Each observable type has its own function for creating observation model settings - in this example we will use the 
`one_way_doppler_instantaneous()` function to model a series of one-way open-loop (i.e. instantaneous) Doppler observations. 
Realise that the individual observation model settings can also include corrective models or define biases for more advanced use-cases.
"""

# Define link ends for two-way Doppler and two-way range observables, for each ground station
link_ends = []
for station in station_names:
    link_ends_per_station = dict()
    link_ends_per_station[observation.transmitter] = observation.body_origin_link_end_id("Orbiter")
    link_ends_per_station[observation.receiver] = observation.body_origin_link_end_id("Orbiter")
    link_ends_per_station[observation.reflector1] = observation.body_reference_point_link_end_id("Enceladus", station)
    link_ends.append(link_ends_per_station)


# Define tracking arcs
# The tracking arcs are (arbitrarily) set to start 2h after the start of each propagation arc.
tracking_arc_duration = 0.9*arc_duration
tracking_arcs_start = []
tracking_arcs_end = []
for arc_start in arc_start_times:
    tracking_arc_start = arc_start + 0.03 * 3600.0
    tracking_arcs_start.append(tracking_arc_start)
    tracking_arcs_end.append(tracking_arc_start + tracking_arc_duration)

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
    observation_settings_list.append(observation.n_way_range(link_definition, [light_time_correction_settings], range_bias_settings))


### Define Observation Simulation Settings
"""
We now have to define the times at which observations are to be simulated. To this end, we will define the settings 
for the simulation of the individual observations from the previously defined observation models. Bear in mind that 
these observation simulation settings are not to be confused with the ones to be used when setting up the estimator 
object, as done just above.

Finally, for each observation model, the observation simulation settings set the times at which observations 
are simulated and defines the viability criteria and noise of the observation.

Note that the actual simulation of the observations requires `Observation Simulators`, which are created automatically 
by the `Estimator` object. Hence, one cannot simulate observations before the creation of an estimator.
"""

# Define observation simulation times for both Doppler and range observables
doppler_cadence = 50.0
range_cadence = 10.0

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
    viability_settings.append(observation.elevation_angle_viability(["Enceladus", station], np.deg2rad(5.0)))
# Check whether Enceladus or Saturn are occulting the signal
    viability_settings.append(observation.body_occultation_viability(["Orbiter", ""], "Enceladus"))
    #viability_settings.append(observation.body_occultation_viability(["Orbiter", ""], "Saturn"))
# Check whether SEP angle is sufficiently large
    #viability_settings.append(observation.body_avoidance_viability(["Orbiter", ""], "Sun", np.deg2rad(15.0)))

# Apply viability checks to all simulated observations
observation.add_viability_check_to_all(observation_simulation_settings, viability_settings)

# Add noise levels of roughly 3.3E-12 [s/m] and add this as Gaussian noise to the observation
doppler_noise = 7.0e-5
range_noise = 0.5 #100.5
#noise_level = 1.0E-3
observation.add_gaussian_noise_to_observable(
    observation_simulation_settings, #[observation_simulation_settings]
    doppler_noise,
    observation.n_way_averaged_doppler_type
    #observation.one_way_instantaneous_doppler_type
)
observation.add_gaussian_noise_to_observable(
    observation_simulation_settings, #[observation_simulation_settings]
    range_noise,
    observation.n_way_range_type
    #observation.one_way_instantaneous_doppler_type
)


## Set up the estimation
"""
Using the defined models for the environment, the propagator, and the observations, we can finally set the actual 
presentation up. In particular, this consists of defining all parameter that should be estimated, the creation 
of the estimator, and the simulation of the observations.

"""

### Defining the parameters to estimate
"""
For this example estimation, we decided to estimate the initial state of `Orbiter`, its drag coefficient, 
and the gravitational parameter of Enceladus. A comprehensive list of parameters available for estimation 
is provided in the FIX LINK.
"""

## Add arc-wise initial states of the Orbiter spacecraft wrt Enceladus
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, arc_start_times)

## Add Enceladus's gravitational parameter. Add estimated parameters to the sensitivity matrix that will be propagated
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Enceladus"))
#parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("Orbiter"))

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


# Create the parameters that will be estimated
#parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings) #, consider_parameters_settings)
estimation_setup.print_parameter_names(parameters_to_estimate)
nb_parameters = len(parameters_to_estimate.parameter_vector)
print("Number of parameters to estimate", len(parameters_to_estimate.parameter_vector))
#print(parameters_to_estimate.parameter_vector[len(parameters_to_estimate.parameter_vector)-9:len(parameters_to_estimate.parameter_vector)-6])
#print(parameters_to_estimate.parameter_vector[len(parameters_to_estimate.parameter_vector)-6:len(parameters_to_estimate.parameter_vector)-3])
#print(parameters_to_estimate.parameter_vector[len(parameters_to_estimate.parameter_vector)-3:len(parameters_to_estimate.parameter_vector)])

k = 0
while k < len(combis_list):
    print('\nk =', k)
    ### Creating the Estimator object
    """
    Ultimately, the `Estimator` object consolidates all relevant information required for the estimation of any system parameter:
        * the environment (bodies)
        * the parameter set (parameters_to_estimate)
        * observation models (observation_settings_list)
        * dynamical, numerical, and integrator setup (propagator_settings)
    
    Underneath its hood, upon creation, the estimator automatically takes care of setting up the relevant 
    Observation Simulator and Variational Equations which will subsequently be required for the simulation 
    of observations and the estimation of parameters, respectively.
    """

    # Create the estimator
    estimator = numerical_simulation.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings)

    ### Perform the observations simulation
    """
    Using the created `Estimator` object, we can perform the simulation of observations by calling its [`simulation_observations()`]
    (https://py.api.tudat.space/en/latest/estimation.html#tudatpy.numerical_simulation.estimation.simulate_observations) 
    function. Note that to know about the time settings for the individual types of observations, this function makes use 
    of the earlier defined observation simulation settings.
    """

    # Simulate required observations
    simulated_observations = estimation.simulate_observations(
        observation_simulation_settings,#    [observation_simulation_settings],
        estimator.observation_simulators,
        bodies)

    # <a id='estimation_section'></a>

    ## Perform the estimation
    """
    Having simulated the observations and created the `Estimator` object - containing the variational equations 
    for the parameters to estimate - we have defined everything to conduct the actual estimation. Realise that up 
    to this point, we have not yet specified whether we want to perform a covariance analysis or the full estimation 
    of all parameters. It should be stressed that the general setup for either path to be followed is entirely identical.
    
    """

    ### Set up the inversion
    """
    To set up the inversion of the problem, we collect all relevant inputs in the form of a estimation input object 
    and define some basic settings of the inversion. Most crucially, this is the step where we can account for 
    different weights - if any - of the different observations, to give the estimator knowledge about the quality 
    of the individual types of observations.
    """

    # Save the true parameters to later analyse the error
    truth_parameters = parameters_to_estimate.parameter_vector

    # Perturb the initial state estimate from the truth (10 m in position; 0.1 m/s in velocity)
    perturbed_parameters = truth_parameters.copy( )
    for i in range(3):
        perturbed_parameters[i] += 10.0
        perturbed_parameters[i+3] += 0.01
    parameters_to_estimate.parameter_vector = perturbed_parameters

    # Create input object for the estimation
    convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=8)
    estimation_input = estimation.EstimationInput(
        simulated_observations,
        convergence_checker=convergence_checker)

    # Set methodological options
    estimation_input.define_estimation_settings(
        reintegrate_variational_equations=False)

    # Define weighting of the observations in the inversion
    weights_per_observable = dict()
    weights_per_observable[observation.n_way_averaged_doppler_type] = doppler_noise ** -2
    weights_per_observable[observation.n_way_range_type] = range_noise ** -2
    #weights_per_observable = {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
    #weights_per_observable = {estimation_setup.observation.n_way_doppler_averaged: noise_level ** -2}
    #weights_per_observable = {estimation_setup.observation.n_way_range: noise_level ** -2}
    estimation_input.set_constant_weight_per_observable(weights_per_observable)

    ### Estimate the individual parameters
    """
    Using the just defined inputs, we can ultimately run the estimation of the selected parameters. After a pre-defined 
    maximum number of iterations (the default value is set to a total of five), the least squares estimator 
    - ideally having reached a sufficient level of convergence - will stop with the process of iterating over the 
    problem and updating the parameters.
    
    Since we have now estimated the actual parameters - unlike when only propagating the covariance matrix over 
    the course of the orbit - we are able to qualitatively compare the goodness-of-fit of the found parameters with the 
    known ground truth ones. Doing this highlights the fact that the formal errors one gets as the result of a covariance 
    analysis tend to sketch a too optimistic version of reality - typically, the true errors are by a certain factor 
    (the true-to-formal-error rate) larger.
    """


    # Perform the estimation
    estimation_output = estimator.perform_estimation(estimation_input)

    # Print the covariance matrix
    #print(estimation_output.formal_errors)
    #print(truth_parameters - parameters_to_estimate.parameter_vector)

    ## Results post-processing
    """
    Finally, to further process the obtained data, one can - exemplary - plot the behaviour of the simulated observations 
    over time, the history of the residuals, or the statistical interpretation of the final residuals.
    
    """

    ### Range-rate over time
    """
    First, we will thus plot all simulations we have simulated over time. One can clearly see how the satellite slowly emerges 
    from the horizont and then more 'quickly' passes the station, until the visibility criterion is not fulfilled anymore.
    """

    """observation_times = np.array(simulated_observations.concatenated_times)
    observations_list = np.array(simulated_observations.concatenated_observations)

    plt.figure(figsize=(9, 5))
    plt.title("Observations as a function of time")
    plt.scatter(observation_times / 3600.0, observations_list)

    plt.xlabel("Time [hr]")
    plt.ylabel("Range rate [m/s]")
    plt.grid()

    plt.tight_layout()
    plt.show()"""

    ### Residuals history
    """
    One might also opt to instead plot the behaviour of the residuals per iteration of the estimator. To this end, we have 
    thus plotted the residuals of the individual observations as a function of time. Note that we can observe a seemingly 
    equal spread around zero. As expected - since we have not defined it this way - the observation is thus not biased.
    """

    """residual_history = estimation_output.residual_history
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
    subplots_list = [ax1, ax2, ax3, ax4]
    
    for i in range(4):
        subplots_list[i].scatter(observation_times, residual_history[:, i])
        subplots_list[i].set_ylabel("Observation Residual [m/s]")
        subplots_list[i].set_title("Iteration "+str(i+1))
    
    
    ax3.set_xlabel("Time since J2000 [s]")
    ax4.set_xlabel("Time since J2000 [s]")
    
    
    plt.tight_layout()
    plt.show()
    """

    ### Final residuals
    """
    Finally, one can also analyse the residuals of the last iteration. Hence, for each of the estimated parameters, 
    we have calculated the true-to-formal-error rate, as well as plotted the statistical distribution of the final 
    residuals between the simulated observations and the estimated orbit. Ideally, given the type of observable we 
    have used (i.e. free of any bias) as well as a statistically sufficient high number of observations, we would 
    expect to see a Gaussian distribution with zero mean here.
    """

    """print('True-to-formal-error ratio:')
    print('\nInitial state')
    print((estimation_output.formal_errors)[:6])
    print(((truth_parameters - parameters_to_estimate.parameter_vector) / estimation_output.formal_errors)[:6])"""
    print('Truth parameters')
    print((truth_parameters)[:6])
    print('Parameters to estimate')
    print((parameters_to_estimate.parameter_vector)[:6])
    """print((truth_parameters-parameters_to_estimate.parameter_vector)[:6])

    print('\nPhysical parameters')
    print(((truth_parameters - parameters_to_estimate.parameter_vector) / estimation_output.formal_errors)[6:])"""
    """
    final_residuals = estimation_output.final_residuals
    
    plt.figure(figsize=(9,5))
    plt.hist(final_residuals, 25)
    plt.xlabel('Final iteration range-rate residual [m/s]')
    plt.ylabel('Occurences [-]')
    plt.title('Histogram of residuals on final iteration')
    
    plt.tight_layout()
    plt.show()
    """

    # Retrieve Doppler observation times for the first arc
    """sorted_observations = simulated_observations.sorted_observation_sets
    doppler_obs_times_s1_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s2_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s3_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][2][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s4_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][3][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s5_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][4][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s6_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][5][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s7_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][6][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s8_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][7][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s9_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][8][0].observation_times if t <= start_gco+arc_duration]
    doppler_obs_times_s10_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_averaged_doppler_type][9][0].observation_times if t <= start_gco+arc_duration]

    range_obs_times_s1_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][0][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s2_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][1][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s3_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][2][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s4_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][3][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s5_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][4][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s6_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][5][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s7_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][6][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s8_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][7][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s9_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][8][0].observation_times if t <= start_gco+arc_duration]
    range_obs_times_s10_first_arc = [(t-start_gco)/3600.0 for t in sorted_observations[observation.n_way_range_type][9][0].observation_times if t <= start_gco+arc_duration]

    #print("np.shape(sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times)", np.shape(sorted_observations[observation.n_way_averaged_doppler_type][0][0].observation_times))
    #print("np.shape(sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times)", np.shape(sorted_observations[observation.n_way_averaged_doppler_type][1][0].observation_times))
    #print("np.shape(sorted_observations[observation.n_way_averaged_doppler_type][2][0].observation_times)", np.shape(sorted_observations[observation.n_way_averaged_doppler_type][2][0].observation_times))
    #print("doppler_obs_times_s1_first_arc", doppler_obs_times_s1_first_arc)
    #print("range_obs_times_s1_first_arc", range_obs_times_s1_first_arc)

    # Plot observation times
    plt.ioff()
    fig = plt.figure(dpi=500)
    # plt.plot(doppler_obs_times_new_forcia_first_arc, np.ones((len(doppler_obs_times_new_forcia_first_arc),1 )))
    # plt.plot(doppler_obs_times_cebreros_first_arc, 2.0 * np.ones((len(doppler_obs_times_cebreros_first_arc),1 )))
    #ax = fig.add_subplot(111)
    #ax.plot(longitude, latitude, '.', markersize=1, color='blue', fillstyle='full')
    plt.scatter(range_obs_times_s1_first_arc, 0.75 * np.ones((len(range_obs_times_s1_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s1_first_arc, 1.0 * np.ones((len(doppler_obs_times_s1_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s2_first_arc, 1.75 * np.ones((len(range_obs_times_s2_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s2_first_arc, 2.0 * np.ones((len(doppler_obs_times_s2_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s3_first_arc, 2.75 * np.ones((len(range_obs_times_s3_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s3_first_arc, 3.0 * np.ones((len(doppler_obs_times_s3_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s4_first_arc, 3.75 * np.ones((len(range_obs_times_s4_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s4_first_arc, 4.0 * np.ones((len(doppler_obs_times_s4_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s5_first_arc, 4.75 * np.ones((len(range_obs_times_s5_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s5_first_arc, 5.0 * np.ones((len(doppler_obs_times_s5_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s6_first_arc, 5.75 * np.ones((len(range_obs_times_s6_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s6_first_arc, 6.0 * np.ones((len(doppler_obs_times_s6_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s7_first_arc, 6.75 * np.ones((len(range_obs_times_s7_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s7_first_arc, 7.0 * np.ones((len(doppler_obs_times_s7_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s8_first_arc, 7.75 * np.ones((len(range_obs_times_s8_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s8_first_arc, 8.0 * np.ones((len(doppler_obs_times_s8_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s9_first_arc, 8.75 * np.ones((len(range_obs_times_s9_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s9_first_arc, 9.0 * np.ones((len(doppler_obs_times_s9_first_arc),1 )), color='red', s=10)
    plt.scatter(range_obs_times_s10_first_arc, 9.75 * np.ones((len(range_obs_times_s10_first_arc),1 )), color='red', s=10)
    #plt.scatter(doppler_obs_times_s10_first_arc, 10.0 * np.ones((len(doppler_obs_times_s10_first_arc),1 )), color='red', s=10)
    plt.xlabel('Observation times [h]')
    plt.ylabel('')
    plt.yticks([0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75], ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'])
    #plt.yticks([0.75, 1, 1.75, 2, 2.75, 3, 3.75, 4, 4.75, 5, 5.75, 6, 6.75, 7, 7.75, 8, 8.75, 9, 9.75, 10], ['S1 R', 'S1 D', 'S2 R', 'S2 D', 'S3 R', 'S3 D', 'S4 R', 'S4 D', 'S5 R', 'S5 D', 'S6 R', 'S6 D', 'S7 R', 'S7 D', 'S8 R', 'S8 D', 'S9 R', 'S9 D', 'S10 R', 'S10 D'])
    plt.ylim([0.25, 10.25])
    plt.title('Viable Doppler and range observations times')
    plt.grid()
    plt.show()"""


    print('norm', np.linalg.norm((truth_parameters - parameters_to_estimate.parameter_vector)[0:3]))
    temporary = np.linalg.norm((truth_parameters - parameters_to_estimate.parameter_vector)[0:3])
    normdiff.append(temporary)

    """del link_ends
    del link_ends_per_station
    del tracking_arc_duration
    del tracking_arcs_start
    del tracking_arcs_end
    del light_time_correction_settings
    del range_bias_settings
    del observation_settings_list
    del link_definition
    del observation_times_doppler
    del observation_times_range
    del time
    del observation_times_per_type
    del observation_simulation_settings
    del viability_settings
    del station_names
    del station_coordinates
    del parameter_settings
    del propagator_settings_list
    del acc_components
    del nb_parameters
    del propagator_settings
    del initial_states
    del integrator_settings
    del arc_start_times
    del arc_end_times
    del arc_start
    del rotation_matrix
    del rotation_matrix_back
    del acceleration_models
    del acceleration_settings
    del accelerations_settings_orbiter
    del nb_arcs
    del arc_duration
    del reference_area
    del drag_coefficient
    del aero_coefficient_settings
    del radiation_pressure_coefficient
    del occulting_bodies
    del radiation_pressure_settings
    del start_gco
    del end_gco
    del global_frame_origin
    del global_frame_orientation
    del bodies_to_propagate
    del central_bodies
    del bodies_to_create
    del body_settings
    del get_gravity_enceladus
    del saturn_gravitational_parameter
    del saturn_reference_radius
    del nor_sh_sat
    del saturn_normalized_cosine_coefficients
    del saturn_normalized_sine_coefficients
    del saturn_associated_reference_frame
    del mu_saturn
    del initial_state_enceladus
    del keplerian_state_enceladus
    del rotation_rate_enceladus
    del initial_orientation_enceladus
    del empty_ephemeris_dict
    del orbiter_ephemeris
    del bodies
    del estimation_input
    del estimation_output
    del truth_parameters
    del parameters_to_estimate"""
    #del observation_times
    #del observations_list
    #del radial_empirical_acceleration_component
    #del along_track_empirical_acceleration_component
    #del across_track_empirical_acceleration_component
    #del propagation_setup
    #del propagator
    #del multi_arc
    k += 1

print(normdiff)
