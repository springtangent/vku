#version 450

/*
To calculate an SDF:

arrange all the shapes in an array so each node's parent goes after it.

iterate through the array, storing the sdf in a parallel array, and using the values in the
array to calculate booleans, for example.

The last value should be the root.
*/

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 frag_color;

#define PI 3.1415926535897932384626433832795

const float infinity = 1. / 0.;


struct AffineTransformation
{
    mat3 m;
    vec3 t;
};

struct Camera
{
    float field_of_view;
    float aspect_ratio;
    AffineTransformation transform;
};

layout(push_constant) uniform PushConstants {
    int material_count;
    int light_count;
    int shape_count;
    Camera camera;
} push_constants;

const int SHAPE_TYPE_SPHERE = 1;
const int SHAPE_TYPE_CAPSULE = 2;
const int SHAPE_TYPE_LINE = 3;
const int SHAPE_TYPE_UNION = 4;
const int SHAPE_TYPE_DIFFERENCE = 5;
const int SHAPE_TYPE_INTERSECTION = 6;

struct ShapeStruct
{
    int type;
    int index_a; // these can be either a child shape index or a material index.
    int index_b;
    float r;
    vec3 a;
    vec3 b;
};

layout(set = 0, binding = 0) buffer ShapeBuffer {
    Shape shapes[];
} shape_buffer;

const int DIRECTIONAL = 0;
const int POINT = 1;
const int SPOT = 2;

struct LightStruct
{
    int type;

    float range;
    float inner_radius;
    float outer_radius;

    vec3 p;
    vec3 d;
    vec3 intensity;
};

layout(set = 0, binding = 1) buffer LightBuffer {
    Light lights[];
} light_buffer;

struct Material
{
    float metallic;
    float roughness;
    vec3 base_color;
    vec3 emissive;
};

layout(set = 0, binding = 2) buffer MaterialBuffer {
    Material materials[];
} material_buffer;


float distribution_ggx(vec3 n, vec3 h, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float ndot_h = max(dot(n, h), 0.0);
    float ndot_h2 = ndot_h * ndot_h;
    float denom = (ndot_h2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2/denom;
}


float geometry_schlick_ggx(float ndot_v, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    float denom = ndot_v * (1.0 - k) + k;

    return ndot_v / denom;
}


float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness)
{
    float ndot_v = max(dot(n, v), 0.0);
    float ndot_l = max(dot(n, l), 0.0);
    float ggx2 = geometry_schlick_ggx(ndot_v, roughness);
    float ggx1 = geometry_schlick_ggx(ndot_l, roughness);

    return ggx1 * ggx2;
}


vec3 fresnel_schlick(float cos_theta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}


struct DirectionalLight
{
    vec3 direction;
    vec3 intensity;
};


struct PointLight
{
    vec3 position;
    vec3 intensity;
    float range;
};


void directional_light_direction_intensity(in vec3 p, in vec3 in_direction, in vec3 in_intensity, out vec3 out_direction, out vec3 out_intensity)
{
    out_direction = normalize(in_direction);
    out_intensity = in_intensity;
}

DirectionalLight directional_light = DirectionalLight(vec3(0.0,-1.0,0.0), vec3(0.8));


void point_light_direction_intensity(in vec3 p, in vec3 position, in vec3 intensity, in float range, out vec3 result_direction, out vec3 result_intensity)
{
    result_direction = normalize(position - p);
    float light_distance = distance(position, p);
    result_intensity = intensity * 1.0 / (1.0 + light_distance / range);
}

PointLight point_light = PointLight(vec3(-2.0, 2.0, 6.0), vec3(1.0), infinity);

void get_sphere(in Shape shape, out vec3 c, out float r)
{
    c = shape.a;
    r = shape.r;
}


float sphere_sdf(in vec3 p, in vec3 c, in float r)
{
    return distance(p, c) - r; 
}


void get_capsule(in Shape shape, out vec3 a, out vec3 b, out float r)
{
    a = shape.a;
    b = shape.b;
    r = shape.r;
}


float capsule_sdf(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 ba = b - a;
    vec3 pa = p - a;
    float t = dot(pa, ba)/dot(ba,ba);
    t = clamp(t, 0.0, 1.0);
    vec3 closest = a + t * ba;
    return distance(p, closest) - r;
}


void get_boolean(in Shape shape, out int sdf_a, out int sdf_b)
{
    sdf_a = shape.index_a;
    sdf_b = shape.index_b;
}

float union_sdf(float a, float b)
{
    return min(a, b);
}

float difference_sdf(float a, float b)
{
    return min(a, -b);
}


float intersection_sdf(float a, float b)
{
    return max(a, b);
}




Material jade = Material(vec3(0.1, 0.8, 0.3), 0.0, 0.5, vec3(0.0, 0.0, 0.0));


Material blend(Material a, Material b, float t) {
    vec3 blended_base_color = mix(a.base_color, b.base_color, t);
    float blended_metallic = (1.0 - t) * a.metallic + t * b.metallic;
    float blended_roughness = (1.0 - t) * a.roughness + t * b.roughness;
    vec3 blended_emissive = mix(a.emissive, b.emissive, t);
    return Material(blended_base_color, blended_metallic, blended_roughness, blended_emissive);
}

Camera camera = Camera(radians(45.0), 640.0/480.0);

struct Config
{
    int max_steps;
    float max_distance;
    float min_step_size;
    float max_step_size;
};

Config config = Config(1000, 10.0, 1e-4, 1.0);

struct Sphere
{
    vec3 c;
    float r;
};

Sphere sphere = Sphere(vec3(0.0, 0.0, 5.0), 1.0);

float sphere_sdf(in vec3 p, in Sphere s)
{
    return distance(p, s.c) - s.r; 
}

float scene(vec3 p)
{
    return sphere_sdf(p, sphere);
}

float trace_ray(in vec3 ray_origin, in vec3 ray_direction)
{
    float t = 0.0;
    int steps = 0;

    while(t < config.max_distance && steps < config.max_steps)
    {
        vec3 p = ray_origin + ray_direction * t;
        float distance_to_surface = scene(p);

        if(distance_to_surface < config.min_step_size)
        {
            return t;
        }

        steps += 1;
        t += min(distance_to_surface, config.max_step_size);      
    }

    return config.max_distance;
}

vec3 sdf_normal(in vec3 point)
{
    // calculate gradient of SDF at point
    float epsilon = 0.001;
    float gradient_x = (scene(vec3(point.x + epsilon, point.y, point.z)) - scene(vec3(point.x - epsilon, point.y, point.z))) / (2.0 * epsilon);
    float gradient_y = (scene(vec3(point.x, point.y + epsilon, point.z)) - scene(vec3(point.x, point.y - epsilon, point.z))) / (2.0 * epsilon);
    float gradient_z = (scene(vec3(point.x, point.y, point.z + epsilon)) - scene(vec3(point.x, point.y, point.z - epsilon))) / (2.0 * epsilon);
    //  calculate normal from gradient
    return normalize(vec3(-gradient_x, -gradient_y, -gradient_z));
}


vec3 shade(vec3 p, Material material, vec3 view_direction, vec3 normal)
{
    vec3 final_color = material.emissive;
    vec3 N = normalize(normal);
    vec3 V = normalize(view_direction);
    vec3 F0 = mix(vec3(0.04), material.base_color, material.metallic);

    vec3 Lo = vec3(0.0);

    // accumulate light
    /*
    vec3 light_direction;
    vec3 light_intensity;
    directional_light_direction_intensity(p, directional_light.direction, directional_light.intensity, light_direction, light_intensity);
    */

    vec3 light_direction;
    vec3 light_intensity;
    point_light_direction_intensity(p, point_light.position, point_light.intensity, point_light.range, light_direction, light_intensity);

    vec3 radiance = light_intensity;
    vec3 H = normalize(light_direction + V);

    // cook-torrance BRDS
    float NDF = distribution_ggx(N, H, material.roughness);
    float G = geometry_smith(N, V, light_direction, material.roughness);
    vec3 F = fresnel_schlick(clamp(dot(N, V), 0.0, 1.0), F0);

    vec3 numerator = NDF * G * F;
    float denomiator = 4.0 * max(dot(N,V), 0.0) * max(dot(N,light_direction), 0.0) + 0.0001;
    vec3 specular = numerator / denomiator;
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;

    float NdotL = max(dot(N, light_direction), 0.0);
    Lo += (kD * material.base_color / PI + specular) * radiance * NdotL;

    // gamma correction.
    vec3 color = Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    return color;
}


void main()
{
    vec2 pixel_pos = 2.0 * uv - 1.0;
    vec3 ray_direction = vec3(pixel_pos * vec2(1.0, -1.0) * vec2(camera.aspect_ratio, 1.0) * tan(camera.field_of_view/2.0), 1.0);
    ray_direction = normalize(ray_direction);
    vec3 ray_origin = vec3(0.0);

    float distance = trace_ray(ray_origin, ray_direction);

    if(distance < config.max_distance)
    {
        // TODO: directional_light
        vec3 p = ray_origin + ray_direction * distance;
        vec3 n = sdf_normal(p);
        frag_color = vec4(shade(p, jade, ray_direction, n), 1.0);
    }
    else
    {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
