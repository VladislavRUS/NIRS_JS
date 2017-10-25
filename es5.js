'use strict';

var Jimp = require('jimp');
//const clustering = require('density-clustering');
var kmeans = require('node-kmeans');
var hmm = require('nodehmm');
var equals = require('array-equal');
var quantizeVertices = require('quantize-vertices');
var fs = require('fs');
var dct = require('dct');

var CLUSTER_NUMBER = 5;
var RESIZE_WIDTH = 256;
var RESIZE_HEIGHT = RESIZE_WIDTH / 2;
var TRESHOLD = 230;
var WRONG = false;

var model = new hmm.Model();

var trainingSetPath = './signatures/trainingSet/';
var rightSetPath = './signatures/rightSet/';
var wrongSetPath = './signatures/wrongSet/';

var trainingSet = fs.readdirSync(trainingSetPath).map(function(file) {
    return trainingSetPath + file;
});
var rightSet = fs.readdirSync(rightSetPath).map(function(file) {
    return rightSetPath + file;
});
var wrongSet = fs.readdirSync(wrongSetPath).map(function(file) {
    return wrongSetPath + file;
});

function start() {
    var vectors = [];
    var allMap = {};

    trainingSet.forEach(function(fileName) {
        console.log('Getting image feature vectors...');

        getImageFeatureVectors(fileName).then(function(dctMap) {

            console.log('Got image feature vectors!');

            var _iteratorNormalCompletion = true;
            var _didIteratorError = false;
            var _iteratorError = undefined;

            try {
                for (var _iterator = dctMap.keys()[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
                    var state = _step.value;


                    if (!allMap[state]) {
                        allMap[state] = [];
                    }

                    allMap[state].push(dctMap.get(state));
                    vectors.push(dctMap.get(state));
                }
            } catch (err) {
                _didIteratorError = true;
                _iteratorError = err;
            } finally {
                try {
                    if (!_iteratorNormalCompletion && _iterator.return) {
                        _iterator.return();
                    }
                } finally {
                    if (_didIteratorError) {
                        throw _iteratorError;
                    }
                }
            }

            if (vectors.length === trainingSet.length * 4) {

                doClustering(vectors).then(function(codeBook) {
                    doHMM(codeBook, dctMap, vectors, allMap);
                });
            }
        });
    });
}

function doClustering(vectors) {

    return new Promise(function(resolve, reject) {
        console.log('Start clustring...');

        kmeans.clusterize(vectors, { k: CLUSTER_NUMBER }, function(err, res) {

            if (err) {
                console.error(err);
                reject(err);
            } else {

                var clusters = res.map(function(cluster, idx) {
                    return {
                        idx: idx,
                        array: cluster.clusterInd
                    };
                });

                var sorted = clusters.map(function(clusterObj) {
                    return { idx: clusterObj.idx, value: clusterObj.array[0] };
                }).sort(function(a, b) {
                    return a.value - b.value;
                });

                var cnt = 0;

                var codeBook = new Map();
                sorted.forEach(function(sortedCluster, idx) {
                    var cluster = clusters.filter(function(cluster) {
                        return sortedCluster.idx === cluster.idx;
                    })[0];
                    codeBook.set(idx, cluster.array);

                    cnt += cluster.array.length;
                });

                resolve(codeBook);
            }
        });
    });
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

    var states = ['S1', 'S2', 'S3', 'S4'];
    model.setStatesSize(states.length);
    model.setStartProbability(PI);

    var emissionProbabilityMatrix = [];
    for (var i = 0; i < states.length; i++) {
        emissionProbabilityMatrix[i] = [];
    }

    var emissionCodeBook = new Map();
    var _iteratorNormalCompletion2 = true;
    var _didIteratorError2 = false;
    var _iteratorError2 = undefined;

    try {
        for (var _iterator2 = codeBook.keys()[Symbol.iterator](), _step2; !(_iteratorNormalCompletion2 = (_step2 = _iterator2.next()).done); _iteratorNormalCompletion2 = true) {
            var code = _step2.value;

            emissionCodeBook.set(code, codeBook.get(code).map(function(idx) {
                return vectors[idx];
            }));
        }
    } catch (err) {
        _didIteratorError2 = true;
        _iteratorError2 = err;
    } finally {
        try {
            if (!_iteratorNormalCompletion2 && _iterator2.return) {
                _iterator2.return();
            }
        } finally {
            if (_didIteratorError2) {
                throw _iteratorError2;
            }
        }
    }

    for (var _i = 0; _i < states.length; _i++) {
        for (var j = 0; j < codeBook.size; j++) {
            emissionProbabilityMatrix[_i][j] = getEmissionProbability(allMap, emissionCodeBook, _i, j);
        }
    }

    var B = emissionProbabilityMatrix;

    console.log(B);

    model.setTransitionProbability(A);
    model.setEmissionProbability(B);

    var vectorsBackup = vectors;

    var right = 0;
    var wrong = 0;

    var checkSet = WRONG ? wrongSet : rightSet;

    checkSet.forEach(function(file) {
        getImageFeatureVectors(file).then(function(dctMap) {

            var imageVectors = [];

            var _iteratorNormalCompletion3 = true;
            var _didIteratorError3 = false;
            var _iteratorError3 = undefined;

            try {
                for (var _iterator3 = dctMap.keys()[Symbol.iterator](), _step3; !(_iteratorNormalCompletion3 = (_step3 = _iterator3.next()).done); _iteratorNormalCompletion3 = true) {
                    var state = _step3.value;

                    imageVectors.push(dctMap.get(state));
                }
            } catch (err) {
                _didIteratorError3 = true;
                _iteratorError3 = err;
            } finally {
                try {
                    if (!_iteratorNormalCompletion3 && _iterator3.return) {
                        _iterator3.return();
                    }
                } finally {
                    if (_didIteratorError3) {
                        throw _iteratorError3;
                    }
                }
            }

            var vectors = vectorsBackup.slice().concat(imageVectors);

            doClustering(vectors).then(function(codeBook) {
                var observationVector = getObservationVector(imageVectors, vectors, codeBook);

                console.log('Observation: ', observationVector);

                var alpha = [];
                for (var _j = 0; _j < states.length; _j++) {
                    alpha[_j] = [];
                }

                var forward = hmm.forward(model, observationVector, alpha);
                var backward = hmm.backward(model, observationVector, alpha);
                console.log(forward, backward);

                if (forward > 0) {
                    right++;
                } else {
                    wrong++;
                }

                console.log('Right: ' + right + '. Wrong: ' + wrong);
            });
        });
    });
}

function toBinary(image, treshold) {
    var black = 0x00000000;
    var white = 0xFFFFFFFF;

    for (var y = 0; y < image.bitmap.height; y++) {
        for (var x = 0; x < image.bitmap.width; x++) {
            var pixelValue = image.getPixelColor(x, y);
            var rgbValue = Jimp.intToRGBA(pixelValue);

            var average = (rgbValue.r + rgbValue.g + rgbValue.b) / 3;

            image.setPixelColor(average > treshold ? white : black, x, y);
        }
    }
}

function getPixelDencity(image) {
    var cnt = 0;

    for (var i = 0; i < image.bitmap.height; i++) {
        for (var j = 0; j < image.bitmap.width; j++) {
            var colorValue = image.getPixelColor(i, j);
            var pixelValue = Jimp.intToRGBA(colorValue);

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
    var alphaU = u === 0 ? 1 / Math.sqrt(M) : Math.sqrt(2 / M);
    var alphaV = v === 0 ? 1 / Math.sqrt(N) : Math.sqrt(2 / N);

    var sum = 0;

    for (var i = 0; i < M; i++) {
        for (var j = 0; j < N; j++) {
            var colorValue = Jimp.intToRGBA(image.getPixelColor(i, j));

            if (colorValue.r > 0) {
                var firstCos = Math.cos((2 * i + 1) * u * Math.PI / (2 * M));
                var secondCos = Math.cos((2 * j + 1) * v * Math.PI / (2 * N));

                sum += firstCos * secondCos;
            }
        }
    }

    return alphaU * alphaV * sum;
}

function getImageFeatureVectors(fileName) {
    return new Promise(function(resolve, reject) {
        Jimp.read(fileName, function(err, image) {
            console.log(fileName);

            image.resize(RESIZE_WIDTH, RESIZE_HEIGHT);
            image.greyscale();
            toBinary(image, TRESHOLD);

            //Get 4 states
            var partials = splitImage([image], 4, { splitMode: 'vertical' });

            //Split each state to 16 block (2^4)
            var statesMap = new Map();
            partials.forEach(function(partial, idx) {
                var partialBlocks = getImageBlocks(partial, 4, 'horizontal');
                statesMap.set(idx, partialBlocks);
            });

            console.log('Computing DCT...');

            //Compute dct of each block
            var dctMap = new Map();
            var _iteratorNormalCompletion4 = true;
            var _didIteratorError4 = false;
            var _iteratorError4 = undefined;

            try {
                var _loop = function _loop() {
                    var state = _step4.value;

                    var blocks = statesMap.get(state);

                    var stateDct = [];
                    var firstOnly = true;

                    blocks.forEach(function(block, idx) {
                        var blockDct = dct(getImageSignal(block));
                        //let blockDct = getPixelDencity(block);

                        stateDct.push(blockDct[0]);
                    });

                    dctMap.set(state, stateDct);
                };

                for (var _iterator4 = statesMap.keys()[Symbol.iterator](), _step4; !(_iteratorNormalCompletion4 = (_step4 = _iterator4.next()).done); _iteratorNormalCompletion4 = true) {
                    _loop();
                }
            } catch (err) {
                _didIteratorError4 = true;
                _iteratorError4 = err;
            } finally {
                try {
                    if (!_iteratorNormalCompletion4 && _iterator4.return) {
                        _iterator4.return();
                    }
                } finally {
                    if (_didIteratorError4) {
                        throw _iteratorError4;
                    }
                }
            }

            console.log(fileName, ' processed!');
            resolve(dctMap);
        });
    });
}

function getObservationVector(imageVectors, vectors, codeBook) {

    var observationVector = [];
    var codedVectorsAll = [];

    var _iteratorNormalCompletion5 = true;
    var _didIteratorError5 = false;
    var _iteratorError5 = undefined;

    try {
        var _loop2 = function _loop2() {
            var code = _step5.value;

            var codedVectors = codeBook.get(code);
            codedVectors = codedVectors.map(function(idx) {
                return vectors[idx];
            });

            codedVectors.forEach(function(cv) {
                if (imageVectors.filter(function(iv) {
                        return equals(iv, cv);
                    }).length > 0) {
                    observationVector.push(code);
                }
            });
        };

        for (var _iterator5 = codeBook.keys()[Symbol.iterator](), _step5; !(_iteratorNormalCompletion5 = (_step5 = _iterator5.next()).done); _iteratorNormalCompletion5 = true) {
            _loop2();
        }
    } catch (err) {
        _didIteratorError5 = true;
        _iteratorError5 = err;
    } finally {
        try {
            if (!_iteratorNormalCompletion5 && _iterator5.return) {
                _iterator5.return();
            }
        } finally {
            if (_didIteratorError5) {
                throw _iteratorError5;
            }
        }
    }

    return observationVector;
}

function getEmissionProbability(allMap, codeBook, state, code) {

    var stateVectors = allMap[state];
    var codedVectors = codeBook.get(code);

    var cnt = 0;

    stateVectors.forEach(function(sv) {
        if (codedVectors.filter(function(cv) {
                return equals(sv, cv);
            }).length) cnt++;
    });

    return parseFloat((cnt / stateVectors.length).toFixed(2));
}

function getImageBlocks(image, depth, splitMode) {
    var blocks = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];

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
        var newImages = [];

        images.forEach(function(image, idx) {
            var _ref = [image.bitmap.width, image.bitmap.height],
                width = _ref[0],
                height = _ref[1];


            var imageCenter = getImageCenter(image);

            var first = void 0,
                second = void 0;

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
    var width = image.bitmap.width,
        height = image.bitmap.height,
        totalSum = 0,
        xSum = 0,
        ySum = 0;

    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            var pixelValue = image.getPixelColor(i, j);
            var rgbValue = Jimp.intToRGBA(pixelValue);
            var average = (rgbValue.r + rgbValue.g + rgbValue.b) / 3;

            totalSum += average;
            xSum += average * j;
            ySum += average * i;
        }
    }

    return {
        x: Math.round(xSum / totalSum),
        y: Math.round(ySum / totalSum)
    };
}

function getImageSignal(image) {
    var signal = [];

    for (var i = 0; i < image.bitmap.height; i++) {
        for (var j = 0; j < image.bitmap.width; j++) {
            var rgb = Jimp.intToRGBA(image.getPixelColor(i, j));

            signal.push(rgb.r === 255 ? 1 : 0);
        }
    }

    return signal;
}

function markCenter(image, imageCenter) {
    var xCenter = imageCenter.x;
    var yCenter = imageCenter.y;

    for (var y = yCenter; y < yCenter + 20; y++) {
        for (var x = xCenter; x < xCenter + 20; x++) {
            image.setPixelColor(0xFF0000, x, y);
        }
    }

    image.write('markCenter.jpg');
}