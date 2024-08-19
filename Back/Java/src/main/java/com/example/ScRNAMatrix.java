package com.example;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class ScRNAMatrix {
	private List<String> cellIds;
	private List<Map<Integer,Short>> countsByCell= new ArrayList<Map<Integer,Short>>();
	private List<String> geneIds;
	private List<Map<Integer,Short>> countsByGene = new ArrayList<Map<Integer,Short>>();
	
	// Pruebas para normalización de matrices
	public static void main(String[] args) throws Exception {
		List<List<Integer>> counts = new ArrayList<>();

        // Leer el archivo y construir la lista de listas
        try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                List<Integer> countEntry = new ArrayList<>();
                for (String part : parts) {
                    countEntry.add(Integer.parseInt(part));
                }
                counts.add(countEntry);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

		List<String> cellIds = new ArrayList<>();
		List<String> geneIds = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            cellIds.add("celula" + i);
        }

        for (int i = 0; i < 3; i++) {
            geneIds.add("gen" + i);
        }

		ScRNAMatrix instance = new ScRNAMatrix(cellIds, geneIds, counts);
		int[][] matrix = toMatrix(instance.cellIds, instance.geneIds, instance.countsByCell);

		double [][] matrixNorm = normalizeMatrix(matrix);
		printMatrix(matrixNorm);
	}

	public static void printMatrix(double[][] matrixNorm) {
		for (int i = 0; i < matrixNorm.length; i++){
			System.out.println("\n");
			for (int j = 0; j < matrixNorm[i].length; j ++){
				System.out.print(matrixNorm[i][j] + " ");
			}
		}
	}

	public ScRNAMatrix(List<String> cellIds, List<String> geneIds, List<List<Integer>> counts) {
		super();
		this.cellIds = cellIds;
		this.geneIds = geneIds;
		initializeCounts(cellIds.size(), geneIds.size());
		for(List<Integer> countList:counts) {
			int cellId = countList.get(0);
			int geneId = countList.get(1);
			short count = (short) Math.min(Short.MAX_VALUE, countList.get(2));
			countsByCell.get(cellId).put(geneId, count);
			countsByGene.get(geneId).put(cellId, count);
		}
	}

	public ScRNAMatrix (int [][] fullMatrix) {
		initializeCounts(fullMatrix.length, fullMatrix[0].length);
		for(int i=0;i<fullMatrix.length;i++) {
			Map<Integer,Short> countsCell = countsByCell.get(i);
			for(int j=0;j<fullMatrix[i].length;j++) {
				Map<Integer,Short> countsGene = countsByGene.get(j);
				int value = fullMatrix[i][j];
				if(value==0) continue;
				if(value >Short.MAX_VALUE) value = Short.MAX_VALUE;
				countsCell.put(j,(short) value);
				countsGene.put(i,(short) value);
			}
		}
	}

	public List<String> getCellIds() {
		return cellIds;
	}
	public List<String> getGeneIds() {
		return geneIds;
	}
	public List<Map<Integer, Short>> getCountsByCell(){
		return countsByCell;
	}

	public Map<Integer,Short> getCountsCell(int cellIdx, int minValue) {
		Map<Integer,Short> answer = new TreeMap<Integer, Short>();
		for(Map.Entry<Integer,Short> entry:countsByCell.get(cellIdx).entrySet()) {
			if(entry.getValue()>=minValue) answer.put(entry.getKey(), entry.getValue());
		}
		return answer;
	}
	private void initializeCounts(int numGenes, int numCells) {
		for(int i=0;i<cellIds.size();i++) countsByCell.add(new TreeMap<Integer, Short>());
		for(int i=0;i<geneIds.size();i++) countsByGene.add(new TreeMap<Integer, Short>());	
	}
	
	public void filterGenes() {
		Set<Integer> idxsToRemove = new HashSet<Integer>();
		for(int j=0;j<countsByGene.size();j++) {
			Map<Integer,Short> countsGene= countsByGene.get(j);
			if(countsGene.size()<10) {
				//System.out.println("Removing gene: "+j+" counts: "+countsGene.size());
				idxsToRemove.add(j);
			}
		}
		removeGenes(idxsToRemove);
		System.out.println("Remaining genes: "+countsByGene.size()+" removed: "+idxsToRemove.size());
		
	}

	public void removeGenes(Set<Integer> idxsToRemove) {
		List<String> geneIds2 = new ArrayList<String>();
		List<Map<Integer,Short>> countsByGene2 = new ArrayList<Map<Integer,Short>>();
		int j2 = 0;
		for(int j=0;j<countsByGene.size();j++) {
			boolean b = idxsToRemove.contains(j);
			if(b || j!=j2) {
				//System.out.println("Gene: "+j+" to remove: "+b+" new index: "+j2);
				Map<Integer,Short> countsGene= countsByGene.get(j);
				for(int i:countsGene.keySet()) {
					Map<Integer,Short> countsCell = countsByCell.get(i);
					short count = countsCell.remove(j);
					if(!b) countsCell.put(j2,count);
				}
			}
			if(!b) {
				geneIds2.add(geneIds.get(j));
				countsByGene2.add(countsByGene.get(j));
				j2++;
			}
		}
		geneIds = geneIds2;
		countsByGene = countsByGene2;
	}
	public void filterCells() {
		Set<Integer> idxsToRemove = new HashSet<Integer>();
		for(int i=0;i<countsByCell.size();i++) {
			Map<Integer,Short> countsCell = countsByCell.get(i);
			if(countsCell.size()<50) {
				System.out.println("Removing cell: "+i +" count: "+countsCell.size());
				idxsToRemove.add(i);
			}
		}
		removeCells(idxsToRemove);
		System.out.println("Remaining cells: "+countsByCell.size()+" removed: "+idxsToRemove.size());
	}
	public void removeCells(Set<Integer> idxsToRemove) {
		List<String> cellIds2 = new ArrayList<String>();
		List<Map<Integer,Short>> countsByCell2 = new ArrayList<Map<Integer,Short>>();
		int i2 = 0;
		for(int i=0;i<countsByCell.size();i++) {
			boolean b = idxsToRemove.contains(i); 
			if(b || i!=i2) {
				Map<Integer,Short> countsCell = countsByCell.get(i);
				for(int j:countsCell.keySet()) {
					Map<Integer,Short> countsGene = countsByGene.get(j);
					short count = countsGene.remove(i);
					if(!b) countsGene.put(i2,count);
				}
			}
			if(!b) {
				cellIds2.add(cellIds.get(i));
				countsByCell2.add(countsByCell.get(i));
				i2++;
			}
		}
		cellIds = cellIds2;
		countsByCell = countsByCell2;
	}

	public static int[][] toMatrix(List<String> cellIds,  List<String> geneIds, List<Map<Integer,Short>> countsByCell) {
        int numCells = cellIds.size();
        int numGenes = geneIds.size();
        int[][] matrix = new int[numCells][numGenes];

        for (int i = 0; i < numCells; i++) {
            Map<Integer, Short> counts = countsByCell.get(i);
            for (Map.Entry<Integer, Short> entry : counts.entrySet()) {
                int geneIndex = entry.getKey();
                short count = entry.getValue();
                matrix[i][geneIndex] = count;
            }
        }
        return matrix;
    }

	public static double computeMedian(double[] array){
		double[] copyArray = array.clone();
		Arrays.sort(copyArray);
		double median;
		if (copyArray.length % 2 == 0){
			median = (copyArray[copyArray.length/2] + copyArray[copyArray.length/2 - 1])/2;
		}
		else{
			median = copyArray[copyArray.length/2];
		}
		return median;
	}

	public static double[][] computeMeanStdByRow(double[][] matrix){
		int numRows = matrix.length; 
		int numCols = matrix[0].length;

		double[] variance = new double[numRows];
		double[] mean = new double[numRows];
		double[] std = new double[numRows];

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				mean[i] += matrix[i][j];
			}			
			mean[i] /= numCols;

			for (int j = 0; j < numCols; j++) {
				variance[i] += Math.pow(matrix[i][j] - mean[i], 2);
			}
			variance[i] /= numCols;

			std[i] = Math.sqrt(variance[i]);
		}

		double[][] meanStd = new double [2][numRows];
		meanStd[0] = mean;
		meanStd[1] = std;

		return meanStd;
	}

	public static double[][] normalizeMatrix(int[][] matrix){
		int numRows = matrix.length; 
		int numCols = matrix[0].length;

		// Calcular la suma por columna (células)
		double[] totalSum = new double[numCols];

        for (int col = 0; col < numCols; col++) {
            for (int row = 0; row < numRows; row++) {
                totalSum[col] += matrix[row][col];
            }
        }
		double median = computeMedian(totalSum);

		// Todos los conteos por columnas (células) quedan igual
		double[][] matrixNew = new double[numRows][numCols];
        for (int col = 0; col < numCols; col++) {
            double scalingFactor = median / totalSum[col];
            for (int row = 0; row < numRows; row++) {
                matrixNew[row][col] = matrix[row][col] * scalingFactor;
            }
        }
		
		// Logaritmo (x+1)
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                matrixNew[i][j] = Math.log1p(matrixNew[i][j]);
            }
        }

		// Normalización por filas (genes)
		double[][] meanStd =  computeMeanStdByRow(matrixNew);
		double[] mean = meanStd[0];
		double[] std = meanStd[1];

		for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                matrixNew[i][j] = (matrixNew[i][j] - mean[i]) / std[i];
            }
        }

		return matrixNew;
	}

}