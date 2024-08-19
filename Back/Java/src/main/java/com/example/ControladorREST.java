package com.example;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import lombok.extern.slf4j.Slf4j;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.Iterator;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

@RestController
@Slf4j
public class ControladorREST {

    @GetMapping("/")
    public String comienzo(){
        log.info("Estoy ejecutando el controlador REST");
        log.debug("Mas información");
        return "Hola Mundo en Spring";
    }
    
    //Esto de CrossOrigin arregla los problemas de CORS
    @CrossOrigin
    @PostMapping("/api/greet")
    public ResponseEntity<Map<String,String>> greetUser(@RequestBody GreetRequest request) {
        String name = request.getName();
        Integer age = request.getAge();

        Map<String,String> response = new HashMap<>();
        response.put("message", "Hola, " + name);

        if (age != null){
            response.put("age", "Tienes " + age + " años");
        }
        return ResponseEntity.ok(response);
    }
    
    private List<String> loadIds(String filename) throws IOException {
		List<String> ids = new ArrayList<>();
		try(FileInputStream st1 = new FileInputStream(filename);
			ConcatGZIPInputStream st2 = new ConcatGZIPInputStream(st1);
			BufferedReader in = new BufferedReader(new InputStreamReader(st2))) {
			String line = in.readLine();
			while(line!=null) {
				int i = line.indexOf(" ");
				if(i>0) ids.add(line.substring(0,i));
				else ids.add(line);
				line = in.readLine();
			}
		}
		return ids;
	}
    
    @CrossOrigin
    @PostMapping("/readCsvFile")
    public ResponseEntity<Map<String,String>> readCsvFile(MultipartFile file) {
        Map<String,String> response = new HashMap<>();
        
        String filename = file.getOriginalFilename();
        Path filepath = Paths.get("/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp", filename);
        try{
            Files.copy(file.getInputStream(), filepath, StandardCopyOption.REPLACE_EXISTING);
            List<String> listCsv = loadIds(filepath.toString());
            String content = String.join(", ", listCsv);
            
            response.put("data", content);

            Files.delete(filepath);
            return ResponseEntity.ok(response);
        }
        catch(Exception e){
            response.put("message", e.getMessage());
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(response);
        }
    }  

    private List<List<Integer>> loadMtxMatrix(String directory) throws IOException{
        List<List<Integer>> counts = new ArrayList<List<Integer>>();
		try (CellRangerMatrixFileReader reader = new CellRangerMatrixFileReader(directory+"/matrix.mtx.gz")) {
			Iterator<CellRangerCount> it = reader.iterator();
			while (it.hasNext()) {
				CellRangerCount count = it.next();
				List<Integer> countList = new ArrayList<Integer>();
				countList.add(count.getCellIdx());
				countList.add(count.getGeneIdx());
				countList.add((int) count.getCount());
				counts.add(countList);
			}
		}
        return counts;
    }

    @CrossOrigin
    @PostMapping("/readMtxFile")
    public ResponseEntity<Map<String,String>> readMtxFile(MultipartFile file) {
        Map<String,String> response = new HashMap<>();
        
        String filename = file.getOriginalFilename();
        Path filepath = Paths.get("/Users/melissa/Documents/Documents/Tesis/Desarrollo/sc-KMSTC/Back/upload_temp", filename);
        try{
            Files.copy(file.getInputStream(), filepath, StandardCopyOption.REPLACE_EXISTING);
            List<List<Integer>> counts = loadMtxMatrix(filepath.toString());
            String content = counts.toString();
            
            response.put("data", content);

            Files.delete(filepath);
            return ResponseEntity.ok(response);
        }
        catch(Exception e){
            response.put("message", e.getMessage());
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(response);
        }
    }  
}