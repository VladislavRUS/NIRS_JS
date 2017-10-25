const Jimp = require('jimp');
//const clustering = require('density-clustering');
const kmeans = require('node-kmeans');
const hmm = require('nodehmm');
const equals = require('array-equal');
const quantizeVertices = require('quantize-vertices');
const fs = require('fs');
const dct = require('dct');

const CLUSTER_NUMBER = 5;
const RESIZE_WIDTH = 256;
const RESIZE_HEIGHT = RESIZE_WIDTH / 2;
const TRESHOLD = 230;
const WRONG = false;

let model = new hmm.Model();

const trainingSetPath = './signatures/trainingSet/';
const rightSetPath = './signatures/rightSet/';
const wrongSetPath = './signatures/wrongSet/';

let trainingSet = fs.readdirSync(trainingSetPath).map(file => trainingSetPath + file);
let rightSet = fs.readdirSync(rightSetPath).map(file => rightSetPath + file);
let wrongSet = fs.readdirSync(wrongSetPath).map(file => wrongSetPath + file);

function start() {
    let vectors = [];
    let allMap = {};

    trainingSet.forEach(fileName => {
        console.log('Getting image feature vectors...');

        getImageFeatureVectors(fileName).then(dctMap => {

            console.log('Got image feature vectors!');

            for (let state of dctMap.keys()) {

                if (!allMap[state]) {
                    allMap[state] = [];
                }

                allMap[state].push(dctMap.get(state));
                vectors.push(dctMap.get(state));
            }

            if (vectors.length === trainingSet.length * 4) {

                doClustering(vectors).then(codeBook => {
                    doHMM(codeBook, dctMap, vectors, allMap);
                });
            }
        });
    });

}

function doClustering(vectors) {

    return new Promise((resolve, reject) => {
        console.log('Start clustring...');

        kmeans.clusterize(vectors, { k: CLUSTER_NUMBER }, (err, res) => {

            if (err) {
                console.error(err);
                reject(err);

            } else {

                let clusters = res.map((cluster, idx) => {
                    return {
                        idx: idx,
                        array: cluster.clusterInd
                    }
                });

                let sorted = clusters.map(clusterObj => { return { idx: clusterObj.idx, value: clusterObj.array[0] } }).sort((a, b) => a.value - b.value);

                let cnt = 0;

                let codeBook = new Map();
                sorted.forEach((sortedCluster, idx) => {
                    let cluster = clusters.filter(cluster => sortedCluster.idx === cluster.idx)[0];
                    codeBook.set(idx, cluster.array);

                    cnt += cluster.array.length;
                });

                resolve(codeBook);
            }
        });
    })
}

start();

function doHMM(codeBook, dctMap, vectors, allMap) {
    console.log('Constructing HMM...');

    //Create HMM
    var A = [
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ];

    var PI = [1, 0, 0, 0];

    let states = ['S1', 'S2', 'S3', 'S4'];
    model.setStatesSize(states.length);
    model.setStartProbability(PI);

    let emissionProbabilityMatrix = [];
    for (let i = 0; i < states.length; i++) {
        emissionProbabilityMatrix[i] = [];
    }

    let emissionCodeBook = new Map();
    for (let code of codeBook.keys()) {
        emissionCodeBook.set(code, codeBook.get(code).map(idx => vectors[idx]));
    }

    for (let i = 0; i < states.length; i++) {
        for (let j = 0; j < codeBook.size; j++) {
            emissionProbabilityMatrix[i][j] = getEmissionProbability(allMap, emissionCodeBook, i, j);
        }
    }

    var B = emissionProbabilityMatrix;

    console.log(B);

    model.setTransitionProbability(A);
    model.setEmissionProbability(B);

    const vectorsBackup = vectors;

    let right = 0;
    let wrong = 0;

    let checkSet = WRONG ? wrongSet : rightSet;

    checkSet.forEach(file => {
        getImageFeatureVectors(file).then(dctMap => {

            let imageVectors = [];

            for (let state of dctMap.keys()) {
                imageVectors.push(dctMap.get(state));
            }

            let vectors = vectorsBackup.slice().concat(imageVectors);

            doClustering(vectors).then(codeBook => {
                let observationVector = getObservationVector(imageVectors, vectors, codeBook);

                console.log('Observation: ', observationVector);

                let alpha = [];
                for (let j = 0; j < states.length; j++) {
                    alpha[j] = [];
                }

                var forward = hmm.forward(model, observationVector, alpha);
                var backward = hmm.backward(model, observationVector, alpha);
                console.log(forward, backward);

                if (forward > 0) {
                    right++;

                } else {
                    wrong++;
                }

                console.log(`Right: ${right}. Wrong: ${wrong}`);
            })
        });
    })
}

function toBinary(image, treshold) {
    let black = 0x00000000;
    let white = 0xFFFFFFFF;

    for (let y = 0; y < image.bitmap.height; y++) {
        for (let x = 0; x < image.bitmap.width; x++) {
            let pixelValue = image.getPixelColor(x, y);
            let rgbValue = Jimp.intToRGBA(pixelValue);

            let average = (rgbValue.r + rgbValue.g + rgbValue.b) / 3;

            image.setPixelColor(average > treshold ? white : black, x, y);
        }
    }
}

function getPixelDencity(image) {
    let cnt = 0;

    for (let i = 0; i < image.bitmap.height; i++) {
        for (let j = 0; j < image.bitmap.width; j++) {
            let colorValue = image.getPixelColor(i, j);
            let pixelValue = Jimp.intToRGBA(colorValue);

            if (pixelValue.r === 0) {
                cnt++;
            }
        }
    }

    //return cnt;
    return cnt / (image.bitmap.width * image.bitmap.height);
}

// function dct(image, firstOnly) {

//     let M = image.bitmap.width;
//     let N = image.bitmap.height;

//     let coefficients = [];
//     for (let i = 0; i < M; i++) {
//         coefficients[i] = [];
//     }

//     for (let u = 0; u < M; u++) {
//         for (let v = 0; v < N; v++) {
//             coefficients[u][v] = getCoefficient(u, v, M, N, image);

//             if (firstOnly) {
//                 return coefficients;
//             }
//         }
//     }

//     return coefficients;
// }

function getCoefficient(u, v, M, N, image) {
    let alphaU = u === 0 ? 1 / Math.sqrt(M) : Math.sqrt(2 / M);
    let alphaV = v === 0 ? 1 / Math.sqrt(N) : Math.sqrt(2 / N);

    let sum = 0;

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let colorValue = Jimp.intToRGBA(image.getPixelColor(i, j));

            if (colorValue.r > 0) {
                let firstCos = Math.cos(((2 * i + 1) * u * Math.PI) / (2 * M));
                let secondCos = Math.cos(((2 * j + 1) * v * Math.PI) / (2 * N));

                sum += firstCos * secondCos;
            }
        }
    }

    return alphaU * alphaV * sum;
}

function getImageFeatureVectors(fileName) {
    return new Promise((resolve, reject) => {
        Jimp.read(fileName, (err, image) => {
            console.log(fileName);

            image.resize(RESIZE_WIDTH, RESIZE_HEIGHT);
            image.greyscale();
            toBinary(image, TRESHOLD);

            //Get 4 states
            let partials = splitImage([image], 4, { splitMode: 'vertical' });

            //Split each state to 16 block (2^4)
            let statesMap = new Map();
            partials.forEach((partial, idx) => {
                let partialBlocks = getImageBlocks(partial, 4, 'horizontal');
                statesMap.set(idx, partialBlocks);
            });

            console.log('Computing DCT...');

            //Compute dct of each block
            let dctMap = new Map();
            for (let state of statesMap.keys()) {
                let blocks = statesMap.get(state);

                let stateDct = [];
                let firstOnly = true;

                blocks.forEach((block, idx) => {
                    let blockDct = dct(getImageSignal(block));
                    //let blockDct = getPixelDencity(block);

                    stateDct.push(blockDct[0]);
                });

                dctMap.set(state, stateDct);
            }

            console.log(fileName, ' processed!');
            resolve(dctMap);
        });
    });
}

function getObservationVector(imageVectors, vectors, codeBook) {

    let observationVector = [];
    let codedVectorsAll = [];

    for (let code of codeBook.keys()) {
        let codedVectors = codeBook.get(code);
        codedVectors = codedVectors.map(idx => vectors[idx]);

        codedVectors.forEach(cv => {
            if (imageVectors.filter(iv => equals(iv, cv)).length > 0) {
                observationVector.push(code);
            }
        });
    }

    return observationVector;
}

function getEmissionProbability(allMap, codeBook, state, code) {

    let stateVectors = allMap[state];
    let codedVectors = codeBook.get(code);

    let cnt = 0;

    stateVectors.forEach(sv => {
        if (codedVectors.filter(cv => {
                return equals(sv, cv);
            }).length) cnt++;
    });

    return parseFloat((cnt / stateVectors.length).toFixed(2));
}

function getImageBlocks(image, depth, splitMode, blocks = []) {
    if (blocks.length !== 2 ** depth) {

        if (blocks.length === 0) {
            blocks.push(image);
        };

        blocks = splitImage(blocks, 2, { splitMode: splitMode });

        return getImageBlocks(image, depth, splitMode === 'horizontal' ? 'vertical' : 'horizontal', blocks);

    } else {
        return blocks;
    }
}

function splitImage(images, partsNumber, options) {
    if (!options.initial) {
        options.initial = images.length;
    }

    if (images.length !== options.initial * partsNumber) {
        let newImages = [];

        images.forEach((image, idx) => {
            let [width, height] = [image.bitmap.width, image.bitmap.height];

            let imageCenter = getImageCenter(image);

            let first, second;

            if (options.splitMode === 'vertical') {
                first = image.clone().crop(0, 0, imageCenter.x, height); //left
                second = image.clone().crop(imageCenter.x, 0, width - imageCenter.x, height); //right

            } else if (options.splitMode === 'horizontal') {
                first = image.clone().crop(0, 0, width, imageCenter.y); //bottom
                second = image.clone().crop(0, imageCenter.y, width, height - imageCenter.y); //top

            } else throw new Error('Not supported split type!');

            newImages.push(first, second);
        });

        images = newImages;

        return splitImage(images, partsNumber, options);

    } else return images;
}

function getImageCenter(image) {
    let width = image.bitmap.width,
        height = image.bitmap.height,
        totalSum = 0,
        xSum = 0,
        ySum = 0;

    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            let pixelValue = image.getPixelColor(i, j);
            let rgbValue = Jimp.intToRGBA(pixelValue);
            let average = (rgbValue.r + rgbValue.g + rgbValue.b) / 3;

            totalSum += average;
            xSum += average * j;
            ySum += average * i;
        }
    }

    return {
        x: Math.round(xSum / totalSum),
        y: Math.round(ySum / totalSum)
    }
}

function getImageSignal(image) {
    let signal = [];

    for (let i = 0; i < image.bitmap.height; i++) {
        for (let j = 0; j < image.bitmap.width; j++) {
            let rgb = Jimp.intToRGBA(image.getPixelColor(i, j));

            signal.push(rgb.r === 255 ? 1 : 0);
        }
    }

    return signal;
}

function markCenter(image, imageCenter) {
    let xCenter = imageCenter.x;
    let yCenter = imageCenter.y;

    for (let y = yCenter; y < yCenter + 20; y++) {
        for (let x = xCenter; x < xCenter + 20; x++) {
            image.setPixelColor(0xFF0000, x, y);
        }
    }

    image.write('markCenter.jpg');
}