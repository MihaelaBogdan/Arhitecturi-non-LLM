const genCanvas = document.getElementById('geneticCanvas');
const gctx = genCanvas.getContext('2d');

const BLOCK = 20;
const COLS = genCanvas.width / BLOCK;
const ROWS = genCanvas.height / BLOCK;
const target = { x: Math.floor(COLS/2), y: 3 }; // Food
const startPos = { x: Math.floor(COLS/2), y: ROWS - 3 };

const lifespan = 60;
let lifeP = 0;
let population;
let genCount = 1;
let genInterval;

class DNA {
    constructor(genes) {
        if (genes) {
            this.genes = genes;
        } else {
            this.genes = [];
            for (let i = 0; i < lifespan; i++) {
                this.genes[i] = Math.floor(Math.random() * 4); // 0=U, 1=R, 2=D, 3=L
            }
        }
    }
    crossover(partner) {
        let newgenes = [];
        let mid = Math.floor(Math.random() * this.genes.length);
        for (let i = 0; i < this.genes.length; i++) {
            if (i > mid) newgenes[i] = this.genes[i];
            else newgenes[i] = partner.genes[i];
        }
        return new DNA(newgenes);
    }
    mutation() {
        for (let i = 0; i < this.genes.length; i++) {
            if (Math.random() < 0.05) {
                this.genes[i] = Math.floor(Math.random() * 4);
            }
        }
    }
}

class GeneticSnake {
    constructor(dna) {
        this.pos = { x: startPos.x, y: startPos.y };
        this.dna = dna || new DNA();
        this.fitness = 0;
        this.completed = false;
        this.crashed = false;
        this.path = [{...this.pos}];
    }
    update() {
        if (this.completed || this.crashed) return;

        let move = this.dna.genes[lifeP];
        if (move === 0) this.pos.y -= 1;
        else if (move === 1) this.pos.x += 1;
        else if (move === 2) this.pos.y += 1;
        else if (move === 3) this.pos.x -= 1;

        if (this.pos.x < 0 || this.pos.x >= COLS || this.pos.y < 0 || this.pos.y >= ROWS) {
            this.crashed = true;
            return;
        }

        // Wall obstacle
        if (this.pos.y === Math.floor(ROWS/2) && this.pos.x > 5 && this.pos.x < COLS - 5) {
            this.crashed = true;
            return;
        }

        if (this.pos.x === target.x && this.pos.y === target.y) {
            this.completed = true;
        }

        this.path.push({...this.pos});
        if(this.path.length > 4) this.path.shift();
    }
    show() {
        gctx.fillStyle = this.completed ? '#10b981' : (this.crashed ? 'rgba(239, 68, 68, 0.2)' : 'rgba(59, 130, 246, 0.4)');
        for(let p of this.path) {
            gctx.fillRect(p.x * BLOCK, p.y * BLOCK, BLOCK, BLOCK);
            gctx.strokeStyle = '#0f172a';
            gctx.strokeRect(p.x * BLOCK, p.y * BLOCK, BLOCK, BLOCK);
        }
        gctx.fillStyle = this.completed ? '#10b981' : (this.crashed ? '#ef4444' : '#3b82f6');
        gctx.fillRect(this.pos.x * BLOCK, this.pos.y * BLOCK, BLOCK, BLOCK);
    }
    calcFitness() {
        let d = Math.abs(this.pos.x - target.x) + Math.abs(this.pos.y - target.y);
        this.fitness = 100 / (d + 1);
        if (this.completed) this.fitness *= 10;
        if (this.crashed) this.fitness /= 10;
    }
}

class Population {
    constructor() {
        this.snakes = [];
        this.popsize = 150;
        for (let i = 0; i < this.popsize; i++) {
            this.snakes[i] = new GeneticSnake();
        }
        this.matingPool = [];
    }
    evaluate() {
        let maxfit = 0;
        for (let i = 0; i < this.popsize; i++) {
            this.snakes[i].calcFitness();
            if (this.snakes[i].fitness > maxfit) {
                maxfit = this.snakes[i].fitness;
            }
        }
        for (let i = 0; i < this.popsize; i++) {
            this.snakes[i].fitness /= maxfit;
        }
        this.matingPool = [];
        for (let i = 0; i < this.popsize; i++) {
            let n = this.snakes[i].fitness * 100;
            for (let j = 0; j < n; j++) {
                this.matingPool.push(this.snakes[i]);
            }
        }
    }
    selection() {
        let newSnakes = [];
        for (let i = 0; i < this.snakes.length; i++) {
            let parentA = this.matingPool[Math.floor(Math.random() * this.matingPool.length)].dna;
            let parentB = this.matingPool[Math.floor(Math.random() * this.matingPool.length)].dna;
            let child = parentA.crossover(parentB);
            child.mutation();
            newSnakes[i] = new GeneticSnake(child);
        }
        this.snakes = newSnakes;
    }
    run() {
        for (let i = 0; i < this.popsize; i++) {
            this.snakes[i].update();
            this.snakes[i].show();
        }
    }
}

function drawGenEnv() {
    gctx.fillStyle = '#0f172a';
    gctx.fillRect(0, 0, genCanvas.width, genCanvas.height);
    
    // Food
    gctx.fillStyle = '#ef4444';
    gctx.fillRect(target.x * BLOCK, target.y * BLOCK, BLOCK, BLOCK);
    
    // Wall
    gctx.fillStyle = '#334155';
    for(let x=6; x<COLS-5; x++){
        gctx.fillRect(x * BLOCK, Math.floor(ROWS/2) * BLOCK, BLOCK, BLOCK);
    }
}

function genStep() {
    drawGenEnv();
    population.run();
    lifeP++;
    if (lifeP === lifespan) {
        population.evaluate();
        population.selection();
        lifeP = 0;
        genCount++;
        document.getElementById('gen-count').innerText = genCount;
    }
}

function startEvolution() {
    if (!population) population = new Population();
    if (!genInterval) genInterval = setInterval(genStep, 30);
}

function resetEvolution() {
    clearInterval(genInterval);
    genInterval = null;
    population = new Population();
    lifeP = 0;
    genCount = 1;
    document.getElementById('gen-count').innerText = genCount;
    drawGenEnv();
}

drawGenEnv();
