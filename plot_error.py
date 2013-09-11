
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    
    b2 = open('batches.pkl', 'rb')
    b1 = open('batches1.pkl', 'rb')
    b3 = open('batches3.pkl', 'rb')

    batch2 = cPickle.load(b2)
    batch1 = cPickle.load(b1)
    batch3 = cPickle.load(b3)
    
    e1 = open('errors1.pkl', 'rb')
    e2 = open('errors.pkl', 'rb')
    e3 = open('errors3.pkl', 'rb')


    errors1 = cPickle.load(e1)
    errors2 = cPickle.load(e2)
    errors3 = cPickle.load(e3)
    
    l1 = open('legends1.pkl', 'rb')
    l2 = open('legends.pkl', 'rb')
    l3 = open('legends3.pkl', 'rb')

    
    legends1 = cPickle.load(l1)
    legends2 = cPickle.load(l2)
    legends3 = cPickle.load(l3)

    plt.axis([0, 25000, 0, 0.15])


    p1a = plt.plot(batch1[0], errors1[0])
    p1b = plt.plot(batch1[1], errors1[1])
    

    p2a = plt.plot(batch2[0], errors2[0])
    p2b = plt.plot(batch2[1], errors2[1])
    
    p3a = plt.plot(batch3[0], errors3[0])
    p3b = plt.plot(batch3[1], errors3[1])

    leg = legends1 + legends2 + legends3
    
    #print leg
    
    plt.legend([p1a, p1b, p2a, p2b, p3a, p3b], leg)
    plt.savefig('plot.png')
    
    
    
    
    
if __name__ == '__main__':
    main()