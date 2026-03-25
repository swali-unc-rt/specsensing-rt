#pragma once

#include <stdexcept>

class SignalGenerator {
public:
    SignalGenerator(unsigned int numSignals, unsigned int numSamples, double fs, double fcmin, double fcmax, double fbmin, double fbmax, double Amin, double Amax, double snr_db);
    ~SignalGenerator();

    void setSample(unsigned int idx, double value) {
        if( idx >= numSamples ) throw std::out_of_range("Sample index out of range");
        samples[idx] = value;
    }
    double getSample(unsigned int idx);
    double getSampleUnsafe(unsigned int idx) { return samples[idx]; }
    inline unsigned int getNumSamples() { return numSamples; }
    double* getSamples() { return samples; }
private:
    unsigned int numSamples;
    double* samples;

    unsigned char* genRandomData(unsigned int dataLen);
};