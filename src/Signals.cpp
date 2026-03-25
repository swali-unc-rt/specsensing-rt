#include "Signals.hpp"

#include <stdexcept>
#include <new>
#include <cmath>
#include <valarray>

#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>

double Signal::sampleWithCarrier(double t, double fc, double Ampfloor) {
    unsigned int idx = (unsigned int)(t / getSymbolDuration());
    return Ampfloor + (sin(fc*(t-(idx*getSymbolDuration()))) * sample(t));
}

double QAMSignal::sample(double t) {
    // First, determine which symbol we're on
    unsigned int idx = (unsigned int)(t / getSymbolDuration());

    // Only M bits
    auto input = sampleData(idx) & ( ( 1 << M ) - 1 );
    
    // Consider a grid of size sqrt(M) * sqrt(M), as a simplification, the x/y coordinate is I and Q
    // and each point is separated by "spacing". x= val % sqrt(M), y= (val-x)/sqrt(M)

    unsigned int sqrtM = 1 << ( __builtin_ctz(M) / 2 );
    unsigned int xC = input % (unsigned int)sqrtM;
    unsigned int yC = (input - xC) / sqrtM;

    double I = (xC*spacing) - (spacing*(sqrtM-1)/2.);
    double Q = (yC*spacing) - (spacing*(sqrtM-1)/2.);

    double A = sqrt( I * I + Q * Q );
    double phi = atan2( Q, I );

    return A*sin(f*(t-(idx*getSymbolDuration()))+phi);
}

double PSKSignal::sample(double t) {
    // First, determine which symbol we're on
    unsigned int idx = (unsigned int)(t / getSymbolDuration());

    // Only M bits
    auto input = sampleData(idx) & ( ( 1 << M ) - 1 );

    // Consider the 2pi circle divided into M compartments, that's what we have
    double phi = (2.*M_PI*input/M) + (M_PI/(double)M);

    return A*sin(f*(t-(idx*getSymbolDuration()))+phi);
}

double ASKSignal::sample(double t) {
    // First, determine which symbol we're on
    unsigned int idx = (unsigned int)(t / getSymbolDuration());

    // Only M bits
    auto input = sampleData(idx) & ( ( 1 << M ) - 1 );

    return ((input+1)*spacing)*sin(f*(t-(idx*getSymbolDuration()))+0);
}

QAMSignal::QAMSignal(unsigned int M, double spacing, unsigned char* data, unsigned int dataLen, double f, double symbolDuration)
    : Signal(data, dataLen, symbolDuration) {
    if( spacing < 0 ) throw std::invalid_argument("invalid spacing");
    if( !M ) throw std::invalid_argument("Invalid modulation order");
    if( (M & (M-1)) != 0 ) throw std::invalid_argument("mod order must be a power of 2");

    this->M = M;
    this->spacing = spacing;
    this->f = f;
}

PSKSignal::PSKSignal(unsigned int M, unsigned char* data, unsigned int dataLen, double f, double A, double symbolDuration)
    : Signal(data, dataLen, symbolDuration ) {
    if( !M ) throw std::invalid_argument("Invalid modulation order");
    if( A == 0 ) throw std::invalid_argument("Invalid amplitude");

    this->M = M;
    this->f = f;
    this->A = A;
}

ASKSignal::ASKSignal(unsigned int M, double spacing, unsigned char* data, unsigned int dataLen, double f, double symbolDuration)
    : Signal(data, dataLen, symbolDuration) {
    if( spacing < 0 ) throw std::invalid_argument("invalid spacing");
    if( !M ) throw std::invalid_argument("Invalid modulation order");

    this->M = M;
    this->spacing = spacing;
    this->f = f;
}

Signal::Signal(unsigned char* data, unsigned int dataLen, double symbolDuration) {
    this->data = nullptr;

    if( !data ) throw std::invalid_argument("null data");
    if( !dataLen ) throw std::invalid_argument("invalid data size");
    if( symbolDuration <= 0 ) throw std::invalid_argument("invalid symbolDuration");

    this->data = (unsigned char*)malloc(dataLen * sizeof(data[0]));
    if( !this->data ) throw std::bad_alloc();

    memcpy(this->data, data, dataLen * sizeof(data[0]));
    this->dataLen = dataLen;
    this->symbolDuration = symbolDuration;
}

Signal::~Signal() {
    if( data )
        free( (void*)data );
}

unsigned char Signal::sampleData(unsigned int idx) {
    if( idx >= dataLen ) throw std::out_of_range("bad sample index");
    return sampleDataUnsafe(idx);
}