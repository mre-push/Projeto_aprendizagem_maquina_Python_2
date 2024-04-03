{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c5371a87",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "usage: ipykernel_launcher.py [-h] [-t] [-g] [-f] [-a] [-w] [-s] [-o]\n",
      "ipykernel_launcher.py: error: unrecognized arguments: C:\\Users\\Mateus\\AppData\\Roaming\\jupyter\\runtime\\kernel-1ef96020-e086-4999-8c6c-f2485aa3965a.json\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "2",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Mateus\\anaconda3\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3513: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "import argparse\n",
    "\n",
    "parser = argparse.ArgumentParser(\n",
    "  description='Train and visualize a Convolutional Neural Network for the MNIST database.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-t', '--train', dest='train', action='store_true',\n",
    "  help='WWhen present, plots the activation of a example on the convolutional layers.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-g', '--render-graphs', dest='graphs', action='store_true',\n",
    "  help='When present, creates validation evolution graphs and prints the test results.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-f', '--render-filters', dest='filters', action='store_true',\n",
    "  help='When present, plots the filters trained on the convolutional layers.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-a', '--render-activations', dest='activation', action='store_true',\n",
    "  help='When present, plots the activation of a example on the convolutional layers.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-w', '--render-wrong',dest='wrong', action='store_true',\n",
    "  help='When present, plots a wrong guess with the expected value and the predicted value.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-s', '--show-plots', dest='show', action='store_true',\n",
    "  help='When present, opens many windows with the requested renders.'\n",
    ")\n",
    "parser.add_argument(\n",
    "  '-o', '--output-plots', dest='save_img', action='store_true',\n",
    "  help='When present, saves the requested renders to images on the img folder.'\n",
    ")\n",
    "\n",
    "args = parser.parse_args()\n",
    "\n",
    "# Imports libs after parsing for better help performance\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from cnn import *\n",
    "from functions import *\n",
    "\n",
    "train = args.train\n",
    "graphs = args.graphs\n",
    "filters = args.filters\n",
    "activation = args.activation\n",
    "wrong = args.wrong\n",
    "show = args.show\n",
    "save_img = args.save_img\n",
    "\n",
    "# Extract training data and testing data\n",
    "(X_train, Y_train), (X_test, Y_test) = extract_data('data')\n",
    "\n",
    "# Get all models to test\n",
    "models_params = get_test_models()\n",
    "\n",
    "# If I am to calculate the validation metrics evolution graphs\n",
    "if graphs:\n",
    "  # Decide to train or use already trained models\n",
    "  if train:\n",
    "    # Trains and return metrics data\n",
    "    data = train_models(models_params, X_train, Y_train, X_test, Y_test)\n",
    "  else:\n",
    "    # Loads and return metrics data\n",
    "    data = load_models(models_params, X_test, Y_test)\n",
    "  # Print test results and plot all data\n",
    "  present_metrics(data)\n",
    "\n",
    "# Load the best model for the visualizations\n",
    "model = MnistModel('padrao-4-6-500')\n",
    "model.load_model()\n",
    "\n",
    "if filters:\n",
    "  # Plot all the filters of the model\n",
    "  visualize_filters(model)\n",
    "\n",
    "if activation:\n",
    "  # Plot the outputs of the convolutional layers\n",
    "  visualize_activations(model, X_test[0:1,:,:,:])\n",
    "\n",
    "if wrong:\n",
    "  # Plot some wrong guesses with the expected results and predicted values\n",
    "  visualize_wrong(model, X_test, Y_test)\n",
    "\n",
    "if (graphs or filters or activation or wrong) and save_img:\n",
    "  # Find all figures plotted and save them\n",
    "  save_all_figs()\n",
    "\n",
    "if (graphs or filters or activation or wrong) and show:\n",
    "  # Show every single plot\n",
    "  plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23b943fa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
