//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Adam Zsombor
// Neptun : X079FB
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char *vertexSource = R"(
    #version 330
    precision highp float;

    layout(location = 0) in vec2 cVertexPosition;
    out vec2 texcoord;

    void main() {
        texcoord = (cVertexPosition + vec2(1, 1))/2;
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
    }
)";

const char *fragmentSource = R"(
    #version 330
    precision highp float;

    uniform sampler2D textureUnit;
    in  vec2 texcoord;
    out vec4 fragmentColor;

    void main() {
        fragmentColor = texture(textureUnit, texcoord);
    }
)";

vec3 powVec3(vec3 a, int n) {
    vec3 temp = vec3(1, 1, 1);
    for (int i = 0; i < n; i++) {
        temp = temp * a;
    }
    return temp;
}

vec3 operator/(vec3 a, vec3 b) {
    return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

class Material {
public:
    vec3 ambient, diffuse, specular, F0;
    float shine, ior;
    int type;
    
    Material(vec3 d, vec3 s, float _shine, int t) : ambient(d * 3), diffuse(d), specular(s), shine(_shine), type(t) {}
    
    Material(vec3 eta, vec3 kappa, int t) : type(t) {
        vec3 etaMinus = eta - vec3(1, 1, 1);
        vec3 etaPlus = eta + vec3(1, 1, 1);
        F0 = (powVec3(etaMinus, 2) + powVec3(kappa, 2)) / (powVec3(etaPlus, 2) + powVec3(kappa, 2));
    }
    
    Material(vec3 eta, int t) : type(t) {
        vec3 etaMinus = eta - vec3(1, 1, 1);
        vec3 etaPlus = eta + vec3(1, 1, 1);
        F0 = powVec3(etaMinus, 2) / powVec3(etaPlus, 2);
        ior = eta.z;
    }
    
    vec3 Fresnel(float gamma) {
        return F0 + (vec3(1, 1, 1) - F0) * powf(1 - gamma, 5);
    }
};

class Ray {
public:
    vec3 start, direction;
    bool out;
    
    Ray(vec3 s, vec3 d, bool o) : start(s), direction(normalize(d)), out(o) {}
};

class Hit {
public:
    float t;
    vec3 position, normal;
    Material* material;
    
    Hit() : t(-1) {}
};

class Camera {
    vec3 eye, lookat, right, up;
    float fov;
    
public:
    Camera() : lookat(vec3(0, 0, 0)), fov(45 * M_PI / 180) {}
    
    void set(vec3 _eye) {
        eye = _eye;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vec3(0, 1, 0), w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir, true);
    }
    
    void reset() {
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vec3(0, 1, 0), w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    
    void rotate() {
        float angle = atan2f(eye.z, eye.x);
        angle -= M_PI / 4;
        eye.x = 4 * cosf(angle);
        eye.z = 4 * sinf(angle);
        reset();
    }
};

class Intersectable {
protected:
    Material* material;
    
    vec2 quadEq(float a, float b, float c) {
        float d = powf(b, 2) - 4 * a * c;
        if (d < 0) {
            return vec2(-1, -1);
        }
        
        float x1 = (-b + sqrtf(d)) / (2 * a);
        float x2 = (-b - sqrtf(d)) / (2 * a);
        return vec2(x1, x2);
    }

    float getSmallerT(vec2 ts) {
        if (ts.y < 0) {
            return ts.x;
        } else if (ts.x < 0) {
            return ts.y;
        }
        return fminf(ts.x, ts.y);
    }
    
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

class Plane : public Intersectable {
    Material* tile1, *tile2;
    
    int getNumber(float coord) {
        if (coord >= 0) {
            return 11 + (int)coord;
        }
        return 10 - (int)coord;
    }
    
public:
    Plane(Material* m1, Material* m2) : tile1(m1), tile2(m2) {}
    
    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 P0 = vec3(0, -1, 0);
        hit.normal = vec3(0, 1, 0);
        hit.t = dot((P0 - ray.start), hit.normal) / dot(ray.direction, hit.normal);
        hit.position = ray.start + ray.direction * hit.t;
        
        if (hit.position.x > 10 || hit.position.x < -10 || hit.position.z > 10 || hit.position.z < -10) {
            hit.t = -1;
        }
        
        int x = getNumber(hit.position.x);
        int z = getNumber(hit.position.z);
        
        if ((x % 2 == 0 && z % 2 == 0) || (x % 2 == 1 && z % 2 == 1)) {
            hit.material = tile1;
        } else {
            hit.material = tile2;
        }
        return hit;
    }
};

class Cylinder : public Intersectable {
    vec3 P0, direction;
    float radius, height;
    
public:
    Cylinder(vec3 p, vec3 d, Material* m) : P0(p), direction(normalize(d)), radius(0.3f), height(2) {
        material = m;
    }
    
    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 d = ray.start - P0;
        vec3 X = normalize(cross(direction, vec3(0, 1, 0)));
        vec3 Y = normalize(cross(X, direction));
        vec3 newRayStart = vec3(dot(d, X), dot(d, direction), dot(d, Y));
        vec3 newRayDirection = vec3(dot(ray.direction, X), dot(ray.direction, direction), dot(ray.direction, Y));
        
        float a = powf(newRayDirection.x, 2) + powf(newRayDirection.z, 2);
        float b = 2 * (newRayStart.x * newRayDirection.x + newRayStart.z * newRayDirection.z);
        float c = powf(newRayStart.x, 2) + powf(newRayStart.z, 2) - powf(radius, 2);
        
        vec2 result = quadEq(a, b, c);
        if (result.x <= 0 && result.y <= 0) {
            return hit;
        }
        
        hit.t = getSmallerT(result);
        hit.position = ray.start + ray.direction * hit.t;
        float temp = dot(hit.position - P0, direction);
        if (temp > height || temp < 0) {
            hit.t = -1;
            return hit;
        }
        
        hit.normal = normalize(hit.position - P0 - direction * temp);
        hit.material = material;
        return hit;
    }
};

class Cone : public Intersectable {
    vec3 P0, direction;
    float alpha, height;
    
public:
    Cone(vec3 p, vec3 d, Material* m) : P0(p), direction(normalize(d)), alpha(0.2f), height(2) {
        material = m;
    }
    
    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 d = ray.start - P0;
        
        float a = powf(dot(ray.direction, direction), 2) - powf(cosf(alpha), 2) * dot(ray.direction, ray.direction);
        float b = 2 * (dot(ray.direction, direction) * dot(d, direction) - powf(cosf(alpha), 2) * dot(d, ray.direction));
        float c = powf(dot(d, direction), 2) - dot(d, d) * powf(cosf(alpha), 2);
        
        vec2 result = quadEq(a, b, c);
        if (result.x <= 0 && result.y <= 0) {
            return hit;
        }
        
        hit.t = getSmallerT(result);
        hit.position = ray.start + ray.direction * hit.t;
        float temp = dot(hit.position - P0, direction);
        if (temp > height || temp < 0) {
            hit.t = -1;
            return hit;
        }
        
        hit.normal = normalize(2 * dot(hit.position - P0, direction) * direction - 2 * (hit.position - P0) * powf(cosf(alpha), 2));
        hit.material = material;
        return hit;
    }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 d, vec3 l) : direction(normalize(d)) , Le(l) {}
};

class World {
    std::vector<Intersectable*> objects;
    Light* light;
    Camera* camera;
    vec3 La;
    float epsilon;
    
public:
    World() {
        camera = new Camera();
        camera->set(vec3(0, 1, 4));

        La = vec3(0.4f, 0.4f, 0.4f);
        light = new Light(vec3(1, 1, 1), vec3(2, 2, 2));
        epsilon = 0.0001f;

        Material* white = new Material(vec3(0.3f, 0.3f, 0.3f), vec3(0, 0, 0), 0, 0);
        Material* blue = new Material(vec3(0, 0.1f, 0.3f), vec3(0, 0, 0), 0, 0);
        objects.push_back(new Plane(white, blue));
        
        Material* yellow = new Material(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2), 50, 0);
        Cylinder* c1 = new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1f), yellow);
        objects.push_back(c1);
        
        Material* gold = new Material(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f), 1);
        Cylinder* c2 = new Cylinder(vec3(1, -1, 0), vec3(0.1f, 1, 0), gold);
        objects.push_back(c2);
        
        Material* water = new Material(vec3(1.3f, 1.3f, 1.3f), 2);
        Cylinder* c3 = new Cylinder(vec3(0, -1, -0.8f), vec3(-0.2f, 1, -0.1f), water);
        objects.push_back(c3);
        
        Material* cyan = new Material(vec3(0.1f, 0.2f, 0.3f), vec3(2, 2, 2), 100, 0);
        Cone* c4 = new Cone(vec3(0, 1, 0), vec3(-0.1f, -1, -0.05f), cyan);
        objects.push_back(c4);
        
        Material* magenta = new Material(vec3(0.3f, 0, 0.2f), vec3(2, 2, 2), 20, 0);
        Cone* c5 = new Cone(vec3(0, 1, 0.8f), vec3(0.2f, -1, 0), magenta);
        objects.push_back(c5);
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera->getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable* object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
                bestHit = hit;
            }
        }
        if (dot(ray.direction, bestHit.normal) > 0) {
            bestHit.normal = -bestHit.normal;
        }
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {
        for (Intersectable * object : objects) {
            if (object->intersect(ray).t > 0) {
                return true;
            }
        }
        return false;
    }
    
    vec3 roughMaterialTrace(Hit hit, Ray ray) {
        vec3 out = hit.material->ambient * La;
        Ray shadowRay(hit.position + hit.normal * epsilon, light->direction, true);
        float alpha = dot(hit.normal, light->direction);
        
        if (alpha > 0 && !shadowIntersect(shadowRay)) {
            out = out + light->Le * hit.material->diffuse * alpha;
            vec3 halfway = normalize(-ray.direction + light->direction);
            float beta = dot(hit.normal, halfway);
            
            if (beta > 0) {
                out = out + light->Le * hit.material->specular * powf(beta, hit.material->shine);
            }
        }
        return out;
    }
    
    vec3 reflectMaterialTrace(Hit hit, Ray ray, int depth) {
        vec3 out = vec3(0, 0, 0);
        float gamma = -dot(hit.normal, ray.direction);
        vec3 Fresnel = hit.material->Fresnel(gamma);
        vec3 ref = ray.direction - hit.normal * 2 * dot(hit.normal, ray.direction);
        out = out + trace(Ray(hit.position + hit.normal * epsilon, ref, true), depth + 1) * Fresnel;
        return out;
    }
    
    vec3 refractMaterialTrace(vec3 out, Hit hit, Ray ray, int depth) {
        float gamma = -dot(hit.normal, ray.direction);
        vec3 Fresnel = hit.material->Fresnel(gamma);
        float ior = (ray.out) ? hit.material->ior : 1 / hit.material->ior;
        float temp = 1 - (1 - powf(gamma, 2)) / powf(ior, 2);
        if (temp >= 0) {
            vec3 refractedDir = ray.direction / ior + hit.normal * (gamma / ior - sqrtf(temp));
            out = out + trace(Ray(hit.position - hit.normal * epsilon, refractedDir, false), depth + 1) * (vec3(1, 1, 1) - Fresnel);
        }
        return out;
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > 5) {
            return La;
        }
        
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) {
            return La;
        }
        
        vec3 out;
        
        if (hit.material->type == 0) {
            out = roughMaterialTrace(hit, ray);
        }
        
        if (hit.material->type >= 1) {
            out = reflectMaterialTrace(hit, ray, depth);
        }
        
        if (hit.material->type > 1) {
            out = out + refractMaterialTrace(out, hit, ray, depth);
        }
        
        return out;
    }
    
    void rotate() {
        camera->rotate();
    }
};

GPUProgram gpuProgram;

class WindowTexture {
    unsigned int vao, textureId;
    
public:
    WindowTexture() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    
    void loadTexture(std::vector<vec4>& image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }

    void Draw() {
        glBindVertexArray(vao);
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

World* world;
WindowTexture* window;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    world = new World();
    window = new WindowTexture();
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    std::vector<vec4> image(windowWidth * windowHeight);
    world->render(image);
    window->loadTexture(image);
    window->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'a') {
        world->rotate();
        glutPostRedisplay();
    }
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}
