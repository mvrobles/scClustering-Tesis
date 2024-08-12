package src;

public class WeightedEdge {
	int v1;
	int v2;
	int weight;
	public WeightedEdge(int v1, int v2, int weight) {
		super();
		this.v1 = v1;
		this.v2 = v2;
		this.weight = weight;
	}
	public void addWeight(int w) {
		//weight+=weight;
		weight+=w; // MELI
	}
	public int getV1() {
		return v1;
	}
	public int getV2() {
		return v2;
	}
	public int getWeight() {
		return weight;
	}
	
}