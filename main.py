from scr import *
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def main():
    print('START')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        # interpretation_percentile_estimator()
        #kernel_example()
        #example_1()
        #example_2()
        #example_3()
        #example_3_wf()
        # example_3_table()
        # real_data_example()
        # real_data_example_table()
        # real_data_example2()
        #assess_calibration_model1()



    assess_calibration_model1()
    print('END')

if __name__ == '__main__':
    main()


