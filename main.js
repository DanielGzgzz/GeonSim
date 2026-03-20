// Topological Geon Physics Engine

// Initialize Three.js scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 50;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// -----------------------------------------------------------------------------
// Core Physics Constants & Vacuum Fluid Material
// -----------------------------------------------------------------------------

// Planck's constant as kinematic action viscosity (arbitrary scale for viz)
const PLANCK_H = 6.626e-34;
const VISCOSITY_SCALE = 1.0;

// Speed of light strictly internal wave velocity
const SPEED_OF_LIGHT = 299792458; // c

const vacuumFluidUniforms = {
    uTime: { value: 0.0 },
    uPlanckH: { value: PLANCK_H * VISCOSITY_SCALE },
    uSpeedOfLight: { value: SPEED_OF_LIGHT }
};

const vacuumFluidVertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;

    void main() {
        vUv = uv;
        vPosition = position;
        vNormal = normal;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const vacuumFluidFragmentShader = `
    uniform float uTime;
    uniform float uPlanckH;
    uniform float uSpeedOfLight;

    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;

    void main() {
        // Zero-point tensor fluid active fluctuations
        // Simulated via simple sine waves based on time and position for visualization
        float fluctuation = sin(vPosition.x * 10.0 + uTime * 5.0) *
                            cos(vPosition.y * 10.0 + uTime * 5.0) * 0.5 + 0.5;

        // Base color representing vacuum energy density
        vec3 vacuumColor = vec3(0.05, 0.1, 0.2);
        vec3 activeColor = vec3(0.1, 0.3, 0.5) * fluctuation;

        gl_FragColor = vec4(vacuumColor + activeColor, 0.8);
    }
`;

const vacuumMaterial = new THREE.ShaderMaterial({
    uniforms: vacuumFluidUniforms,
    vertexShader: vacuumFluidVertexShader,
    fragmentShader: vacuumFluidFragmentShader,
    transparent: true,
    side: THREE.DoubleSide
});

// Basic light setup
const ambientLight = new THREE.AmbientLight(0x404040); // Soft white light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(10, 10, 10).normalize();
scene.add(directionalLight);

// Render vacuum fluid backdrop
const vacuumGeometry = new THREE.PlaneGeometry(200, 200, 64, 64);
const vacuumPlane = new THREE.Mesh(vacuumGeometry, vacuumMaterial);
vacuumPlane.position.z = -50;
scene.add(vacuumPlane);

// -----------------------------------------------------------------------------
// Particle Topologies (Geons)
// -----------------------------------------------------------------------------

// Parametric Geometry requires an explicit function
// For Three.js r128, ParametricGeometry is available but sometimes in examples.
// We will construct standard BufferGeometry using parametric calculations.

function createMobiusDoubleLoopGeometry(radius, tube, radialSegments, tubularSegments) {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const indices = [];

    for (let i = 0; i <= tubularSegments; i++) {
        const u = (i / tubularSegments) * 2 * Math.PI * 2; // 4pi double loop

        for (let j = 0; j <= radialSegments; j++) {
            const v = (j / radialSegments) * 2 * Math.PI;

            // Mobius transform
            const x = (radius + tube * Math.cos(v / 2)) * Math.cos(u);
            const y = (radius + tube * Math.cos(v / 2)) * Math.sin(u);
            const z = tube * Math.sin(v / 2);

            vertices.push(x, y, z);
        }
    }

    // Generate indices
    for (let i = 0; i < tubularSegments; i++) {
        for (let j = 0; j < radialSegments; j++) {
            const a = i * (radialSegments + 1) + j;
            const b = (i + 1) * (radialSegments + 1) + j;
            const c = (i + 1) * (radialSegments + 1) + (j + 1);
            const d = i * (radialSegments + 1) + (j + 1);

            indices.push(a, b, d);
            indices.push(b, c, d);
        }
    }

    geometry.setIndex(indices);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.computeVertexNormals();
    return geometry;
}

function createTrefoilKnotGeometry(radius, tube, tubularSegments, radialSegments, p, q) {
    // A standard TorusKnotGeometry works well for (3,2) trefoil knots (proton)
    return new THREE.TorusKnotGeometry(radius, tube, tubularSegments, radialSegments, p, q);
}

// Visual Representation (Mass Mechanics Balance)
// Base materials
const electronMaterial = new THREE.MeshPhongMaterial({
    color: 0x00ffff,
    wireframe: true,
    emissive: 0x002222,
    shininess: 100
});

const protonMaterial = new THREE.MeshPhongMaterial({
    color: 0xff00ff,
    wireframe: true,
    emissive: 0x220022,
    shininess: 100
});

// Electron: 4pi Möbius double-loop
const electronGeometry = createMobiusDoubleLoopGeometry(5, 1.5, 16, 128);
const electronMesh = new THREE.Mesh(electronGeometry, electronMaterial);
electronMesh.position.set(-10, 0, 0);
scene.add(electronMesh);

// Proton: (3,2)-trefoil knot
const protonGeometry = createTrefoilKnotGeometry(6, 2, 128, 16, 3, 2);
const protonMesh = new THREE.Mesh(protonGeometry, protonMaterial);
protonMesh.position.set(10, 0, 0);
scene.add(protonMesh);


// -----------------------------------------------------------------------------
// Arrow Helpers for Visualizations
// -----------------------------------------------------------------------------
function createRadialVectors(mesh, color, count, directionSign, length) {
    const helpers = [];
    for (let i = 0; i < count; i++) {
        const dir = new THREE.Vector3(
            Math.cos(i / count * Math.PI * 2),
            Math.sin(i / count * Math.PI * 2),
            0
        ).normalize().multiplyScalar(directionSign);
        const origin = new THREE.Vector3(0, 0, 0); // Local to mesh
        const arrow = new THREE.ArrowHelper(dir, origin, length, color, 0.5, 0.5);
        helpers.push(arrow);
        mesh.add(arrow);
    }
    return helpers;
}

// Visualizing forces directly attached to particles
// Yellow: Casimir Vacuum Pressure (Inward)
// Red: Centrifugal Momentum (Outward)
// Magenta: Transverse/Radial Electric Vectors
const electronCasimirArrows = createRadialVectors(electronMesh, 0xffff00, 8, -1, 4.0);
const protonCasimirArrows = createRadialVectors(protonMesh, 0xffff00, 8, -1, 5.0);

const electronCentrifArrows = createRadialVectors(electronMesh, 0xff0000, 8, 1, 4.0);
const protonCentrifArrows = createRadialVectors(protonMesh, 0xff0000, 8, 1, 5.0);

const electronElectricArrows = createRadialVectors(electronMesh, 0xff00ff, 12, -1, 6.0);
const protonElectricArrows = createRadialVectors(protonMesh, 0xff00ff, 12, 1, 6.0);

// Global Arrows for Coulomb interaction (Gauss shell projection)
const coulombArrowElectron = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), electronMesh.position, 1, 0x00ffff, 1, 1);
const coulombArrowProton = new THREE.ArrowHelper(new THREE.Vector3(-1,0,0), protonMesh.position, 1, 0x00ffff, 1, 1);
scene.add(coulombArrowElectron);
scene.add(coulombArrowProton);


// DOM Elements for UI
const valH = document.getElementById('val-h');
const valC = document.getElementById('val-c');
const valDist = document.getElementById('val-dist');
const valCasimir = document.getElementById('val-casimir');
const valShadow = document.getElementById('val-shadow');
const valCentrifugal = document.getElementById('val-centrifugal');
const valStretch = document.getElementById('val-stretch');
const valCharge = document.getElementById('val-charge');

// Initialize Constants on UI
if(valH) valH.textContent = PLANCK_H.toExponential(3) + ' J⋅s';
if(valC) valC.textContent = SPEED_OF_LIGHT.toExponential(3) + ' m/s';

// Handle window resize
window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

// Render loop setup (to be expanded)
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const time = clock.getElapsedTime();

    // Update vacuum fluid uniforms
    vacuumFluidUniforms.uTime.value = time;

    // -------------------------------------------------------------------------
    // Relativistic Kinematics & Electromagnetism Update
    // -------------------------------------------------------------------------

    // 1. Internal wave propagation strictly fixed to c (simulated rotation)
    // The visual rotation speed is a proxy for the internal light speed looping.
    const internalSpeedProxy = 1.0;

    // Electron rotation (4pi loop)
    electronMesh.rotation.y = time * internalSpeedProxy;
    electronMesh.rotation.x = time * (internalSpeedProxy * 0.5);

    // Proton rotation (trefoil knot)
    protonMesh.rotation.y = -time * internalSpeedProxy;
    protonMesh.rotation.z = time * (internalSpeedProxy * 0.3);

    // 2. Gravitational Attraction (Macroscopic Geometric Shadowing of Casimir Pressure)
    // Calculate distance between electron and proton knots
    const distanceVector = new THREE.Vector3().subVectors(protonMesh.position, electronMesh.position);
    const distanceSq = distanceVector.lengthSq();
    const distance = Math.sqrt(distanceSq);

    // Casimir shadow pressure proxy: inverse square of distance, modified by knot geometry cross-sections
    // Approximation: larger geometry = more shadowing = stronger pull
    const casimirShadowForce = (5.0 * 6.0) / (distanceSq + 0.1);
    const attractionForce = distanceVector.normalize().multiplyScalar(casimirShadowForce * 0.05);

    // 3. Inertial Resistance (Helical Stretching)
    // When subjected to acceleration (attractionForce), the track stretches helically,
    // increasing resistance to further acceleration.
    // Here we apply a simple damping factor proportional to the speed.

    // Applying force to positions
    electronMesh.position.add(attractionForce);
    protonMesh.position.sub(attractionForce); // equal and opposite

    // 4. Electromagnetism (Gauss Shell Illusion via RMS Radial Vectors)
    // Effective charge is calculated by integrating RMS of internal radial vectors.
    // Transverse vectors destructively interfere.
    // The surviving inward radial flux mathematically projects the Gauss shell.
    // Here we simulate the resulting Coulomb force mathematically.

    const coulombForceMagnitude = (1.0 * 1.0) / (distanceSq + 0.1); // Proportional proxy
    const coulombForce = distanceVector.clone().multiplyScalar(coulombForceMagnitude * 0.1);

    // Apply Coulomb force (opposite charges attract)
    electronMesh.position.add(coulombForce);
    protonMesh.position.sub(coulombForce);

    // Keep particles from merging completely for visualization purposes
    if (distance < 12.0) {
        const repulsion = distanceVector.clone().multiplyScalar(-0.5);
        electronMesh.position.add(repulsion);
        protonMesh.position.sub(repulsion);
    }

    // Update Coulomb Arrow Visuals (Cyan)
    const coulombDir = distanceVector.clone().normalize();
    coulombArrowElectron.position.copy(electronMesh.position);
    coulombArrowElectron.setDirection(coulombDir);
    coulombArrowElectron.setLength(Math.max(1, coulombForceMagnitude * 50));

    coulombArrowProton.position.copy(protonMesh.position);
    coulombArrowProton.setDirection(coulombDir.clone().negate());
    coulombArrowProton.setLength(Math.max(1, coulombForceMagnitude * 50));

    // Update UI Panel Real-Time Metrics
    if(valDist) valDist.textContent = distance.toFixed(2);

    // Casimir vacuum pressure vs outward centrifugal momentum balances the "mass"
    const casimirPressure = 100 / (radiusEProxy = 5); // Simplification for UI
    const centrifugalMom = Math.pow(internalSpeedProxy, 2) / 5;

    if(valCasimir) valCasimir.textContent = casimirPressure.toFixed(2) + " units";
    if(valCentrifugal) valCentrifugal.textContent = centrifugalMom.toFixed(2) + " units";

    // Shadowing and inertial resistance
    if(valShadow) valShadow.textContent = casimirShadowForce.toFixed(4) + " units";

    // Inertial helical stretch proxy: proportional to combined acceleration forces
    const acceleration = attractionForce.length() + coulombForce.length();
    if(valStretch) valStretch.textContent = (acceleration * 100).toFixed(4) + " λ";

    // Effective RMS charge integration proxy
    // Root mean square of internal varying radial vectors projecting the Gauss Shell
    const rmsCharge = Math.sqrt(Math.pow(coulombForceMagnitude, 2) / 2);
    if(valCharge) valCharge.textContent = rmsCharge.toFixed(4) + " E_rms";

    renderer.render(scene, camera);
}

// Start simulation
animate();
