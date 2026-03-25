#pragma once

class Signal {
public:
    double sampleWithCarrier(double t, double fc, double Ampfloor);
    virtual double sample(double t) = 0;

    Signal(unsigned char* data, unsigned int dataLen, double symbolDuration);
    virtual ~Signal();

    unsigned char sampleData(unsigned int idx);
    inline unsigned char sampleDataUnsafe(unsigned int idx) { return data[idx]; }
    inline unsigned int getDataLen() { return dataLen; }
    inline double getSymbolDuration() { return symbolDuration; }
    inline bool isValidTime(double t) { return (unsigned int)(t / getSymbolDuration()) < dataLen; }
    inline double getCarrierFrequency() { return fc; }
    inline void setCarrierFrequency(double fc) { this->fc = fc; }
private:
    unsigned char* data;
    unsigned int dataLen;
    double symbolDuration;
    double fc;
};

class ASKSignal : public Signal {
public:
    virtual double sample(double t);

    ASKSignal(unsigned int M, double spacing, unsigned char* data, unsigned int dataLen, double f, double symbolDuration);
private:
    unsigned int M;
    double spacing;
    double f;
};

class QAMSignal : public Signal {
public:
    virtual double sample(double t);

    QAMSignal(unsigned int M, double spacing, unsigned char* data, unsigned int dataLen, double f, double symbolDuration);
private:
    unsigned int M;
    double spacing;
    double f;
};

class PSKSignal : public Signal {
public:
    virtual double sample(double t);

    PSKSignal(unsigned int M, unsigned char* data, unsigned int dataLen, double f, double A, double symbolDuration);
private:
    unsigned int M;
    double f;
    double A;
};