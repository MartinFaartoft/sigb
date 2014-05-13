import numpy as np

def speculate(point, point_normal, light_source, camera_position,alpha):
    #find l
    incident_vector = point - light_source
    incident_vector = incident_vector/np.linalg.norm(incident_vector)
    #find r
    reflection_vector = reflect(point_normal, point, light_source)

    #reflection_vector = 2*np.dot(point_normal,incident_vector)*point_normal - incident_vector
    reflection_vector = reflection_vector/np.linalg.norm(reflection_vector)

    view_vector = camera_position - point
    view_vector = view_vector/np.linalg.norm(view_vector)

    i_s = 1

    i_spectral = i_s*np.dot(view_vector,reflection_vector)**alpha

    return i_spectral

def reflect(normal, point, light_source):
    incident_vector = light_source - point
    #incident_vector = incident_vector/np.linalg.norm(incident_vector)
    return 2 * np.dot(normal,incident_vector) * normal - incident_vector
    
normal = np.array([0,1,0])
point = np.array([0,0,0])
light_source = np.array([-3,3,0])
camera = np.array([4,4.4,0])


print speculate(point, normal, light_source, camera, 100)
print reflect(normal, point, light_source)