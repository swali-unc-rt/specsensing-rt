#include "SignalGenerator.hpp"

#include "Signals.hpp"
#include <stdexcept>
#include <random>
#include <vector>

#include <ctime>

using std::uniform_real_distribution;
using std::default_random_engine;
using std::vector;

double SignalGenerator::getSample(unsigned int idx) {
    if( idx >= numSamples ) throw std::out_of_range("invalid sample index");
    return getSampleUnsafe(idx);
}

unsigned char* SignalGenerator::genRandomData(unsigned int dataLen) {
    unsigned char* data = (unsigned char*)malloc(dataLen);
    if( !data ) throw std::bad_alloc();

    for(unsigned int i = 0; i < dataLen; ++i)
        data[i] = (unsigned char)rand();
    return data;
}

SignalGenerator::SignalGenerator(unsigned int numSignals, unsigned int numSamples, double fs, double fcmin, double fcmax, double fbmin, double fbmax, double Amin, double Amax, double snr_db) {
    samples = nullptr;

    if( numSamples == 0 ) throw std::invalid_argument("numSamples cannot be zero");
    if( numSignals == 0 ) throw std::invalid_argument("numSignals cannot be zero");

    samples = (double*)malloc(numSamples*sizeof(double));
    default_random_engine gen;
    gen.seed((unsigned int)time(nullptr));
    uniform_real_distribution<double> fcarrierSampler(fcmin, fcmax);
    uniform_real_distribution<double> fbasebandSampler(fbmin, fbmax);
    uniform_real_distribution<double> ampSampler(Amin,Amax);
    vector<Signal*> signals;

    for( unsigned int i = 0; i < numSignals; ++i ) {
        // Possible options are: ASKSignal, QAMSignal, PSKSignal
        //  ASKSignal: 2-ASK, 4-ASK
        //  QAMSignal: 16-QAM, 256-QAM
        //  PSKSignal: 4-PSK, 8-PSK
        //

        int sigType = rand() % 6;
        Signal* sig = nullptr;
        auto fb = fbasebandSampler(gen);
        auto fc = fcarrierSampler(gen);
        auto sampleDuration = (double)numSamples / fs;
        unsigned int dataSize = (unsigned int)((sampleDuration * fb)+1);
        auto data = genRandomData(dataSize);

        switch(sigType) {
        case 0: // 2-ASK
            sig = new ASKSignal(2, ampSampler(gen), data, dataSize, fb, 1/fb);
            break;
        case 1: // 4-ASK
            sig = new ASKSignal(4, ampSampler(gen), data, dataSize, fb, 1/fb);
            break;
        case 2: // 16-QAM
            sig = new QAMSignal(16, ampSampler(gen), data, dataSize, fb, 1/fb);
            break;
        case 3: // 256-QAM
            sig = new QAMSignal(256, ampSampler(gen), data, dataSize, fb, 1/fb);
            break;
        case 4: // 4-PSK
            sig = new PSKSignal(4, data, dataSize, fb, ampSampler(gen), 1/fb);
            break;
        case 5: // 8-PSK
            sig = new PSKSignal(8, data, dataSize, fb, ampSampler(gen), 1/fb);
            break;
        default:
            free((void*)data);
            throw std::domain_error("unknown random signal type");
        };

        sig->setCarrierFrequency(fc);
        signals.push_back(sig);

        free((void*)data);
    }

    double signalPower = 0;
    for(unsigned int i = 0; i < numSamples; ++i) {
        samples[i] = 0.;
        for(auto sig : signals) {
            double t = (double)i / fs;
            double fc = sig->getCarrierFrequency();
            samples[i] += sig->sampleWithCarrier(t, fc, 0.);
            signalPower += samples[i] * samples[i];
        }
    }
    signalPower /= (double)numSamples;
    double noisePower = signalPower / std::pow(10., snr_db / 10.);
    double noiseStdDev = std::sqrt(noisePower);

    std::normal_distribution<double> noiseDist(0., noiseStdDev);
    for(unsigned int i = 0; i < numSamples; ++i)
        samples[i] += noiseDist(gen);

    for(auto i : signals)
        delete i;
}

SignalGenerator::~SignalGenerator() {
    if( samples )
        free((void*)samples);
}

