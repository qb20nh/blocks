// --- Data Parsing ---
// 3D array representing the puzzle layers.
// Numbers represent piece IDs. null represents empty space.
const layers = [
    // Layer Z=0
    [
        [1, 1, 2, 2, 3, 3],
        [1, 4, 2, 5, 3, 6],
        [4, 4, 5, 5, 7, 6],
        [null, null, null, 8, 8, 9],
        [null, null, null, 0, 9, 9],
        [null, null, null, 0, 1, 1]
    ],
    // Layer Z=1
    [
        [2, 1, 3, 2, 4, 3],
        [2, 2, 5, 7, 6, 6],
        [4, 6, 5, 7, 7, 7],
        [null, null, null, 8, 8, 8],
        [null, null, null, 8, 9, 1],
        [null, null, null, 0, 0, 1]
    ],
    // Layer Z=2
    [
        [2, 5, 3, 3, 4, 4],
        [6, 5, 5, 3, 7, 4],
        [6, 6, 9, 9, 7, 7],
        [null, null, null, 9, 8, 0],
        [null, null, null, 1, 8, 0],
        [null, null, null, 1, 2, 2]
    ],
    // Layer Z=3
    [
        [null, null, null, null, null, null],
        [null, null, null, null, null, null],
        [null, null, null, null, null, null],
        [null, null, null, 9, 0, 0],
        [null, null, null, 3, 3, 2],
        [null, null, null, 1, 1, 2]
    ],
    // Layer Z=4
    [
        [null, null, null, null, null, null],
        [null, null, null, null, null, null],
        [null, null, null, null, null, null],
        [null, null, null, 4, 4, 5],
        [null, null, null, 3, 6, 5],
        [null, null, null, 3, 7, 7]
    ],
    // Layer Z=5
    [
        [null, null, null, null, null, null],
        [null, null, null, null, null, null],
        [null, null, null, null, null, null],
        [null, null, null, 4, 5, 5],
        [null, null, null, 4, 6, 7],
        [null, null, null, 6, 6, 7]
    ]
];

// --- 3D Scene Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

// 1. Setup Perspective Camera
const perspectiveCamera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
perspectiveCamera.position.set(-7, 10, 10);

// 2. Setup Orthographic Camera
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 12.5;
const orthoCamera = new THREE.OrthographicCamera(
    frustumSize * aspect / -2,
    frustumSize * aspect / 2,
    frustumSize / 2,
    frustumSize / -2,
    0.1,
    1000
);
orthoCamera.position.copy(perspectiveCamera.position);
orthoCamera.lookAt(scene.position);

let activeCamera = perspectiveCamera;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(activeCamera, renderer.domElement);
controls.enableDamping = true;

// --- Lighting ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 1.0);
sunLight.position.set(10, 20, 10);
sunLight.castShadow = false; // Start OFF (Skeleton Mode default ON)
sunLight.shadow.mapSize.width = 4096;
sunLight.shadow.mapSize.height = 4096;
const d = 10;
sunLight.shadow.camera.left = -d;
sunLight.shadow.camera.right = d;
sunLight.shadow.camera.top = d;
sunLight.shadow.camera.bottom = -d;
sunLight.shadow.bias = -0.0005;
scene.add(sunLight);

const pointLight2 = new THREE.PointLight(0xffffff, 0.5);
pointLight2.position.set(-10, -10, -5);
scene.add(pointLight2);

// --- Geometry Processing ---
const colorPalette = [
    "#000000", "#003002", "#6e3f00", "#005a7f", "#007709",
    "#009d9e", "#1cb501", "#01fc52", "#fffe02", "#b8c77a",
    "#ffe8f9", "#11fffa", "#8cbbff", "#fc8aff", "#ff8576",
    "#a78100", "#cd2200", "#ff0094", "#d507ff", "#7b00d4",
    "#0400fc", "#4976ff", "#a26a9e", "#910065", "#490026",
    "#340089", "#010038"
];

const dimZ = layers.length;
const dimY = layers[0].length;
const dimX = layers[0][0].length;

/**
 * Generates a unique string key for a voxel coordinate (x, y, z).
 * Used for Set/Map lookups.
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 * @param {number} z - Z coordinate
 * @returns {string} Key in format "x,y,z"
 */
function getKey(x, y, z) { return `${x},${y},${z}`; }

/**
 * Retrieves the value (piece ID) at a specific voxel coordinate.
 * Returns null if the coordinate is out of the grid bounds.
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 * @param {number} z - Z coordinate
 * @returns {number|null} The value at the coordinate or null if out of bounds.
 */
function getVal(x, y, z) {
    if (z < 0 || z >= dimZ) return null;
    if (y < 0 || y >= dimY) return null;
    if (x < 0 || x >= dimX) return null;
    return layers[z][y][x];
}

/**
 * Returns the 6 orthogonal neighbors of a given voxel.
 * @param {Object} v - Voxel object {x, y, z}
 * @returns {Array<{x:number, y:number, z:number}>} Array of neighbor coordinates.
 */
function getNeighbors(v) {
    return [
        { x: v.x + 1, y: v.y, z: v.z },
        { x: v.x - 1, y: v.y, z: v.z },
        { x: v.x, y: v.y + 1, z: v.z },
        { x: v.x, y: v.y - 1, z: v.z },
        { x: v.x, y: v.y, z: v.z + 1 },
        { x: v.x, y: v.y, z: v.z - 1 },
    ];
}

/**
 * Checks if a set of 4 voxels forms a "Screw" tetracube.
 * A screw tetracube is chiral and has specific pairwise squared distances:
 * [1, 1, 1, 2, 2, 3] (sorted).
 * @param {Array<{x,y,z}>} voxels - Array of 4 voxel objects.
 * @returns {boolean} True if it is a screw tetracube.
 */
function isScrewTetracube(voxels) {
    if (voxels.length !== 4) return false;
    const dists = [];
    // Calculate squared Euclidean distances between all pairs
    for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
            const dx = voxels[i].x - voxels[j].x;
            const dy = voxels[i].y - voxels[j].y;
            const dz = voxels[i].z - voxels[j].z;
            dists.push(dx * dx + dy * dy + dz * dz);
        }
    }
    dists.sort((a, b) => a - b);

    // Expected squared distances for a screw tetracube:
    // 3 edges (dist^2=1), 2 face diagonals (dist^2=2), 1 body diagonal (dist^2=3)
    const expected = [1, 1, 1, 2, 2, 3];
    for (let i = 0; i < 6; i++) {
        if (dists[i] !== expected[i]) return false;
    }
    return true;
}

/**
 * Recursively attempts to partition a set of voxels into valid Screw Tetracubes.
 * Uses backtracking to find a valid combination.
 * @param {Array<{x,y,z}>} voxels - Array of voxels to partition.
 * @returns {Array<Array<{x,y,z}>>|null} Array of tetracubes (each is an array of 4 voxels), or null if impossible.
 */
function partitionIntoScrews(voxels) {
    if (voxels.length === 0) return [];

    // Pick the first voxel and try to find 3 others to form a screw
    const first = voxels[0];
    const others = voxels.slice(1);

    // We need to choose 3 from 'others'
    const combinations = getCombinations(others, 3);

    for (let combo of combinations) {
        const candidate = [first, ...combo];
        if (isScrewTetracube(candidate)) {
            // If valid screw, try to partition the remaining voxels
            const candidateKeys = new Set(candidate.map(v => getKey(v.x, v.y, v.z)));
            const remaining = others.filter(v => !candidateKeys.has(getKey(v.x, v.y, v.z)));

            const res = partitionIntoScrews(remaining);
            if (res !== null) {
                return [candidate, ...res];
            }
        }
    }
    return null; // No valid partition found
}

/**
 * Generates all combinations of k elements from the array.
 * @param {Array} arr - Source array.
 * @param {number} k - Number of elements to choose.
 * @returns {Array<Array>} Array of combinations.
 */
function getCombinations(arr, k) {
    if (k === 0) return [[]];
    if (arr.length === 0) return [];
    const first = arr[0];
    const rest = arr.slice(1);
    // Include first element
    const combsWithFirst = getCombinations(rest, k - 1).map(c => [first, ...c]);
    // Exclude first element
    const combsWithoutFirst = getCombinations(rest, k);
    return [...combsWithFirst, ...combsWithoutFirst];
}

// --- PARSING ---
// Identify connected components (blobs) of same-ID voxels.
// If a blob is larger than 4 voxels (e.g. 8 or 12), partition it into multiple screw tetracubes.
const visited = new Set();
let finalPieces = [];
let nextIdOffset = 10; // Offset to distinguish split pieces from original IDs

for (let z = 0; z < dimZ; z++) {
    for (let y = 0; y < dimY; y++) {
        for (let x = 0; x < dimX; x++) {
            const key = getKey(x, y, z);
            const val = getVal(x, y, z);

            // Start BFS for a new unvisited piece
            if (val !== null && !visited.has(key)) {
                const blobVoxels = [];
                const queue = [{ x, y, z }];
                visited.add(key);

                while (queue.length > 0) {
                    const current = queue.shift();
                    blobVoxels.push(current);
                    const neighbors = getNeighbors(current);
                    for (let n of neighbors) {
                        const nKey = getKey(n.x, n.y, n.z);
                        if (!visited.has(nKey)) {
                            const nVal = getVal(n.x, n.y, n.z);
                            if (nVal === val) {
                                visited.add(nKey);
                                queue.push(n);
                            }
                        }
                    }
                }

                // Process the found blob
                if (blobVoxels.length === 4) {
                    finalPieces.push({ id: val, voxels: blobVoxels });
                } else if (blobVoxels.length % 4 === 0) {
                    // Try to split larger blobs into screws
                    const partitioned = partitionIntoScrews(blobVoxels);
                    if (partitioned) {
                        partitioned.forEach(p => {
                            finalPieces.push({ id: val + nextIdOffset, voxels: p });
                            nextIdOffset += 10;
                        });
                    } else {
                        // Fallback: keep as one large piece if partition fails
                        finalPieces.push({ id: val, voxels: blobVoxels });
                    }
                } else {
                    // Invalid size (not multiple of 4), keep as is
                    finalPieces.push({ id: val, voxels: blobVoxels });
                }
            }
        }
    }
}

// --- TOPOLOGICAL SORT ---
// Determine the assembly order. A piece A "supports" piece B if A is directly below B.
// We build a dependency graph where edge A -> B means A must be placed before B.

finalPieces.forEach((p, idx) => p._index = idx);
const adjList = new Array(finalPieces.length).fill(0).map(() => []);
const inDegree = new Array(finalPieces.length).fill(0);

// Map every voxel to its piece index for O(1) lookup
const voxelMap = new Map();
finalPieces.forEach((p, idx) => {
    p.voxels.forEach(v => {
        voxelMap.set(getKey(v.x, v.y, v.z), idx);
    });
});

// Build the graph by scanning columns
for (let y = 0; y < dimY; y++) {
    for (let x = 0; x < dimX; x++) {
        const columnVoxels = [];
        // Collect all voxels in this vertical column (varying Z)
        for (let z = 0; z < dimZ; z++) {
            const key = getKey(x, y, z);
            if (voxelMap.has(key)) {
                columnVoxels.push({ z: z, pIdx: voxelMap.get(key) });
            }
        }
        // Add dependencies: lower piece supports upper piece
        for (let i = 0; i < columnVoxels.length - 1; i++) {
            const bottom = columnVoxels[i].pIdx;
            const top = columnVoxels[i + 1].pIdx;
            if (bottom !== top) {
                if (!adjList[bottom].includes(top)) {
                    adjList[bottom].push(top);
                    inDegree[top]++;
                }
            }
        }
    }
}

// Perform Kahn's Algorithm for Topological Sort
const sortedIndices = [];
const queue = [];
// Start with pieces that have no dependencies (on the ground)
inDegree.forEach((deg, idx) => { if (deg === 0) queue.push(idx); });

// Heuristic: Sort the queue by lowest Z coordinate to build from bottom up visually
queue.sort((a, b) => {
    const minZA = Math.min(...finalPieces[a].voxels.map(v => v.z));
    const minZB = Math.min(...finalPieces[b].voxels.map(v => v.z));
    return minZA - minZB;
});

while (queue.length > 0) {
    // Re-sort queue to maintain bottom-up visual order among available pieces
    queue.sort((a, b) => {
        const minZA = Math.min(...finalPieces[a].voxels.map(v => v.z));
        const minZB = Math.min(...finalPieces[b].voxels.map(v => v.z));
        return minZA - minZB;
    });
    const u = queue.shift();
    sortedIndices.push(u);

    // Remove dependencies
    adjList[u].forEach(v => {
        inDegree[v]--;
        if (inDegree[v] === 0) {
            queue.push(v);
        }
    });
}

if (sortedIndices.length < finalPieces.length) {
    const used = new Set(sortedIndices);
    const remaining = [];
    for (let i = 0; i < finalPieces.length; i++) if (!used.has(i)) remaining.push(i);
    remaining.sort((a, b) => {
        const minZA = Math.min(...finalPieces[a].voxels.map(v => v.z));
        const minZB = Math.min(...finalPieces[b].voxels.map(v => v.z));
        return minZA - minZB;
    });
    sortedIndices.push(...remaining);
}
const sortedPieces = sortedIndices.map(idx => finalPieces[idx]);

// --- RENDERING OBJECTS ---
const offsetX = 2.5; // Center the model
const boxGeo = new THREE.BoxGeometry(0.95, 0.95, 0.95); // Slightly smaller than 1.0 for gap
const sphereGeo = new THREE.SphereGeometry(0.15, 16, 16); // For skeleton nodes

// Count total voxels
let totalVoxels = 0;
sortedPieces.forEach(p => totalVoxels += p.voxels.length);

// Create Instanced Meshes
// Material for the main block (transparent when skeleton is ON)
const cubeMat = new THREE.MeshStandardMaterial({
    color: 0xffffff, // Will be overridden by instance color
    transparent: true,
    opacity: 0.15,
    depthWrite: false,
    side: THREE.DoubleSide
});
const cubeInstancedMesh = new THREE.InstancedMesh(boxGeo, cubeMat, totalVoxels);
cubeInstancedMesh.castShadow = true;
cubeInstancedMesh.receiveShadow = true;
scene.add(cubeInstancedMesh);

// Material for the skeleton nodes (always opaque)
const sphereMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
const sphereInstancedMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, totalVoxels);
sphereInstancedMesh.castShadow = false;
sphereInstancedMesh.receiveShadow = false;
scene.add(sphereInstancedMesh);

const animPieces = [];
let globalInstanceIdx = 0;
const dummy = new THREE.Object3D();

sortedPieces.forEach((piece, idx) => {
    // Assign a unique color from the palette
    const colorIndex = (idx * 5) % 27;
    const colorHex = colorPalette[colorIndex];
    const color = new THREE.Color(colorHex);

    const pieceGroup = new THREE.Group(); // Still used for lines and logical position
    const instanceStart = globalInstanceIdx;
    const instanceCount = piece.voxels.length;
    const voxelOffsets = [];

    piece.voxels.forEach(v => {
        // Convert grid coordinates to world coordinates
        // Swap Y and Z for Three.js (Y is up)
        const px = v.x - offsetX;
        const py = v.z - 2.5;
        const pz = v.y - offsetX;

        voxelOffsets.push(new THREE.Vector3(px, py, pz));

        // Set Color
        cubeInstancedMesh.setColorAt(globalInstanceIdx, color);
        sphereInstancedMesh.setColorAt(globalInstanceIdx, color);

        // Initial Matrix (Hidden/Scale 0)
        dummy.position.set(px, py, pz);
        dummy.scale.set(0, 0, 0);
        dummy.updateMatrix();
        cubeInstancedMesh.setMatrixAt(globalInstanceIdx, dummy.matrix);
        sphereInstancedMesh.setMatrixAt(globalInstanceIdx, dummy.matrix);

        globalInstanceIdx++;
    });

    // Draw lines between connected voxels in the piece (Skeleton edges)
    // Lines are NOT instanced, they remain in the group
    const linePoints = [];
    for (let i = 0; i < piece.voxels.length; i++) {
        for (let j = i + 1; j < piece.voxels.length; j++) {
            const v1 = piece.voxels[i];
            const v2 = piece.voxels[j];
            const dist = Math.abs(v1.x - v2.x) + Math.abs(v1.y - v2.y) + Math.abs(v1.z - v2.z);
            if (dist === 1) { // Adjacent
                linePoints.push(new THREE.Vector3(v1.x - offsetX, v1.z - 2.5, v1.y - offsetX));
                linePoints.push(new THREE.Vector3(v2.x - offsetX, v2.z - 2.5, v2.y - offsetX));
            }
        }
    }
    if (linePoints.length > 0) {
        const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
        const lineMat = new THREE.LineBasicMaterial({ color: color });
        const lines = new THREE.LineSegments(lineGeo, lineMat);
        lines.userData = { isSkeleton: true };
        pieceGroup.add(lines);
    }

    scene.add(pieceGroup);
    pieceGroup.visible = false; // Start hidden
    pieceGroup.position.y = 20; // Start high up for drop animation

    animPieces.push({
        group: pieceGroup,
        instanceStart: instanceStart,
        instanceCount: instanceCount,
        voxelOffsets: voxelOffsets,
        baseY: 0,
        isLanded: false,
        isFalling: false
    });
});

const axesHelper = new THREE.AxesHelper(4);
scene.add(axesHelper);

// --- UI & CONTROL LOGIC ---
let currentPieceIndex = 0;
const totalPieces = animPieces.length;
const dropSpeed = 0.03;
let isSkeletonVisible = true;

const statusEl = document.getElementById('status');
const nextBtn = document.getElementById('next-btn');
const resetBtn = document.getElementById('reset-btn');
const projBtn = document.getElementById('proj-btn');
const skelBtn = document.getElementById('skel-btn');

function updateUI() {
    statusEl.innerText = `Piece: ${currentPieceIndex} / ${totalPieces}`;
    if (currentPieceIndex >= totalPieces) {
        nextBtn.disabled = true;
        nextBtn.innerText = "Complete";
    } else {
        nextBtn.disabled = false;
        nextBtn.innerText = "Next Block";
    }
}

function resetAnimation() {
    currentPieceIndex = 0;
    animPieces.forEach(p => {
        p.group.visible = false;
        p.group.position.y = 20;
        p.isLanded = false;
        p.isFalling = false;

        // Hide instances
        for (let i = 0; i < p.instanceCount; i++) {
            const idx = p.instanceStart + i;
            const off = p.voxelOffsets[i];
            dummy.position.copy(off);
            dummy.scale.set(0, 0, 0);
            dummy.updateMatrix();
            cubeInstancedMesh.setMatrixAt(idx, dummy.matrix);
            sphereInstancedMesh.setMatrixAt(idx, dummy.matrix);
        }
    });
    cubeInstancedMesh.instanceMatrix.needsUpdate = true;
    sphereInstancedMesh.instanceMatrix.needsUpdate = true;
    updateUI();
}

function triggerNextBlock() {
    if (currentPieceIndex < totalPieces) {
        const p = animPieces[currentPieceIndex];
        p.group.visible = true;
        p.isFalling = true;

        // Show instances (scale 1) at initial position
        for (let i = 0; i < p.instanceCount; i++) {
            const idx = p.instanceStart + i;
            const off = p.voxelOffsets[i];
            dummy.position.set(off.x, off.y + p.group.position.y, off.z);
            dummy.scale.set(1, 1, 1);
            dummy.updateMatrix();
            cubeInstancedMesh.setMatrixAt(idx, dummy.matrix);
            sphereInstancedMesh.setMatrixAt(idx, dummy.matrix);
        }
        cubeInstancedMesh.instanceMatrix.needsUpdate = true;
        sphereInstancedMesh.instanceMatrix.needsUpdate = true;

        currentPieceIndex++;
        updateUI();
    }
}

// Toggle Projection Logic
projBtn.addEventListener('click', () => {
    const prevCamera = activeCamera;

    if (activeCamera === perspectiveCamera) {
        activeCamera = orthoCamera;
        projBtn.innerText = "View: Orthographic";
    } else {
        activeCamera = perspectiveCamera;
        projBtn.innerText = "View: Perspective";
    }

    // Sync physical position and orientation (quaternion) to preserve the exact rotation
    activeCamera.position.copy(prevCamera.position);
    activeCamera.quaternion.copy(prevCamera.quaternion);

    // Update controls to use new camera
    controls.object = activeCamera;
    // No manual lookAt() or update() needed here if copying quaternion; 
    // controls will respect current object state on next frame.
});

// Toggle Skeleton Logic
skelBtn.addEventListener('click', () => {
    isSkeletonVisible = !isSkeletonVisible;
    skelBtn.innerText = isSkeletonVisible ? "Skeleton: ON" : "Skeleton: OFF";

    sunLight.castShadow = !isSkeletonVisible;

    // Toggle Instanced Meshes
    if (isSkeletonVisible) {
        // Transparent Mode
        cubeInstancedMesh.material.transparent = true;
        cubeInstancedMesh.material.opacity = 0.15;
        cubeInstancedMesh.material.depthWrite = false;
        sphereInstancedMesh.visible = true;
    } else {
        // Opaque Mode
        cubeInstancedMesh.material.transparent = false;
        cubeInstancedMesh.material.opacity = 1.0;
        cubeInstancedMesh.material.depthWrite = true;
        sphereInstancedMesh.visible = false;
    }
    cubeInstancedMesh.material.needsUpdate = true;

    // Toggle Lines
    animPieces.forEach(p => {
        p.group.children.forEach(child => {
            if (child.userData.isSkeleton) {
                child.visible = isSkeletonVisible;
            }
        });
    });
});

nextBtn.addEventListener('click', triggerNextBlock);
resetBtn.addEventListener('click', resetAnimation);
updateUI();

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    let dirty = false;

    animPieces.forEach((p) => {
        if (p.isFalling && !p.isLanded) {
            p.group.position.y += (0 - p.group.position.y) * dropSpeed;

            // Update the matrices for this piece's instances
            for (let i = 0; i < p.instanceCount; i++) {
                const globalIdx = p.instanceStart + i;
                const offset = p.voxelOffsets[i];

                // Calculate current position based on falling group Y
                dummy.position.set(
                    offset.x,
                    offset.y + p.group.position.y,
                    offset.z
                );
                dummy.scale.set(1, 1, 1);
                dummy.updateMatrix();

                cubeInstancedMesh.setMatrixAt(globalIdx, dummy.matrix);
                sphereInstancedMesh.setMatrixAt(globalIdx, dummy.matrix);
            }
            dirty = true;

            if (Math.abs(p.group.position.y) < 0.01) {
                p.group.position.y = 0;
                p.isLanded = true;
                // Ensure final position is exact
                for (let i = 0; i < p.instanceCount; i++) {
                    const globalIdx = p.instanceStart + i;
                    const offset = p.voxelOffsets[i];
                    dummy.position.set(offset.x, offset.y, offset.z);
                    dummy.scale.set(1, 1, 1);
                    dummy.updateMatrix();
                    cubeInstancedMesh.setMatrixAt(globalIdx, dummy.matrix);
                    sphereInstancedMesh.setMatrixAt(globalIdx, dummy.matrix);
                }
                dirty = true;
            }
        }
    });

    if (dirty) {
        cubeInstancedMesh.instanceMatrix.needsUpdate = true;
        sphereInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    renderer.render(scene, activeCamera);
}

window.addEventListener('resize', () => {
    const aspect = window.innerWidth / window.innerHeight;

    // Update Perspective
    perspectiveCamera.aspect = aspect;
    perspectiveCamera.updateProjectionMatrix();

    // Update Ortho
    orthoCamera.left = -frustumSize * aspect / 2;
    orthoCamera.right = frustumSize * aspect / 2;
    orthoCamera.top = frustumSize / 2;
    orthoCamera.bottom = -frustumSize / 2;
    orthoCamera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();