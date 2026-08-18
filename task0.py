from linear import generate_linear_data
from visual import plot_2d_graph

def main():
    x,y=generate_linear_data()
    plot_2d_graph(x,y,title="Linearly Separate data")

if __name__=="__main__":
    main()