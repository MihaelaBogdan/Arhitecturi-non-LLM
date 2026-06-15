const astarCanvas = document.getElementById('astarCanvas');
const actx = astarCanvas.getContext('2d');
const cols = 25;
const rows = 25;
const w = astarCanvas.width / cols;
const h = astarCanvas.height / rows;

let grid = new Array(cols);
let openSet = [];
let closedSet = [];
let start;
let end;
let path = [];
let astarInterval;

// Node Class
class Spot {
    constructor(i, j) {
        this.i = i;
        this.j = j;
        this.f = 0;
        this.g = 0;
        this.h = 0;
        this.neighbors = [];
        this.previous = undefined;
        this.wall = false;
    }
    
    show(col) {
        actx.fillStyle = col;
        if(this.wall) {
            actx.fillStyle = '#334155';
            actx.fillRect(this.i*w, this.j*h, w, h);
            actx.strokeStyle = '#0f172a';
            actx.strokeRect(this.i*w, this.j*h, w, h);
        } else {
            actx.fillRect(this.i*w, this.j*h, w-1, h-1);
        }
    }
    
    addNeighbors(grid) {
        let i = this.i;
        let j = this.j;
        if (i < cols - 1) this.neighbors.push(grid[i + 1][j]);
        if (i > 0) this.neighbors.push(grid[i - 1][j]);
        if (j < rows - 1) this.neighbors.push(grid[i][j + 1]);
        if (j > 0) this.neighbors.push(grid[i][j - 1]);
    }
}

function setupAStar() {
    clearInterval(astarInterval);
    for (let i = 0; i < cols; i++) {
        grid[i] = new Array(rows);
        for (let j = 0; j < rows; j++) {
            grid[i][j] = new Spot(i, j);
        }
    }
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            grid[i][j].addNeighbors(grid);
        }
    }
    start = grid[0][0];
    end = grid[cols - 1][rows - 1];
    start.wall = false;
    end.wall = false;
    
    openSet = [start];
    closedSet = [];
    path = [];
    drawAStar();
}

function drawAStar() {
    actx.fillStyle = '#0f172a';
    actx.fillRect(0, 0, astarCanvas.width, astarCanvas.height);
    
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            grid[i][j].show('rgba(255,255,255,0.05)');
        }
    }
    
    for (let i = 0; i < closedSet.length; i++) {
        closedSet[i].show('#ef4444');
    }
    
    for (let i = 0; i < openSet.length; i++) {
        openSet[i].show('#10b981');
    }
    
    path = [];
    let temp = current;
    if(temp) {
        path.push(temp);
        while (temp.previous) {
            path.push(temp.previous);
            temp = temp.previous;
        }
    }
    
    for (let i = 0; i < path.length; i++) {
        path[i].show('#3b82f6');
    }
    
    start.show('#f59e0b');
    end.show('#ec4899');
}

function heuristic(a, b) {
    return Math.abs(a.i - b.i) + Math.abs(a.j - b.j);
}

let current = null;

function stepAStar() {
    if (openSet.length > 0) {
        let winner = 0;
        for (let i = 0; i < openSet.length; i++) {
            if (openSet[i].f < openSet[winner].f) {
                winner = i;
            }
        }
        current = openSet[winner];
        
        if (current === end) {
            clearInterval(astarInterval);
            drawAStar();
            console.log('DONE!');
            return;
        }
        
        openSet.splice(winner, 1);
        closedSet.push(current);
        
        let neighbors = current.neighbors;
        for (let i = 0; i < neighbors.length; i++) {
            let neighbor = neighbors[i];
            if (!closedSet.includes(neighbor) && !neighbor.wall) {
                let tempG = current.g + 1;
                let newPath = false;
                if (openSet.includes(neighbor)) {
                    if (tempG < neighbor.g) {
                        neighbor.g = tempG;
                        newPath = true;
                    }
                } else {
                    neighbor.g = tempG;
                    newPath = true;
                    openSet.push(neighbor);
                }
                
                if (newPath) {
                    neighbor.h = heuristic(neighbor, end);
                    neighbor.f = neighbor.g + neighbor.h;
                    neighbor.previous = current;
                }
            }
        }
    } else {
        console.log('No solution');
        clearInterval(astarInterval);
        return;
    }
    drawAStar();
}

function startAStar() {
    openSet = [start];
    closedSet = [];
    path = [];
    current = start;
    // reset previous
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            grid[i][j].previous = undefined;
            grid[i][j].f = 0;
            grid[i][j].g = 0;
            grid[i][j].h = 0;
        }
    }
    clearInterval(astarInterval);
    astarInterval = setInterval(stepAStar, 20);
}

function resetAStar() {
    setupAStar();
}

let isDrawingWall = false;
astarCanvas.addEventListener('mousedown', () => isDrawingWall = true);
astarCanvas.addEventListener('mouseup', () => isDrawingWall = false);
astarCanvas.addEventListener('mousemove', (e) => {
    if(isDrawingWall) {
        let rect = astarCanvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        let i = Math.floor(x / w);
        let j = Math.floor(y / h);
        if(i>=0 && i<cols && j>=0 && j<rows) {
            if(grid[i][j] !== start && grid[i][j] !== end) {
                grid[i][j].wall = true;
                drawAStar();
            }
        }
    }
});

setupAStar();
