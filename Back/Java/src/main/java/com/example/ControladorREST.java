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
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
        System.out.println(request);
        String name = request.getName();
        Integer age = request.getAge();

        System.out.println(name+age);
        Map<String,String> response = new HashMap<>();
        response.put("message", "Hola, " + name);

        if (age != null){
            response.put("age", "Tienes " + age + " años");
        }
        return ResponseEntity.ok(response);
    }
    
    // private List<String> loadIds(MultipartFile file) throws IOException {
	// 	List<String> ids = new ArrayList<String>();
	// 	try(FileInputStream st1 = (FileInputStream) file.getInputStream();
	// 		ConcatGZIPInputStream st2 = new ConcatGZIPInputStream(st1);
	// 		BufferedReader in = new BufferedReader(new InputStreamReader(st2))) {
	// 		String line = in.readLine();
	// 		while(line!=null) {
	// 			int i = line.indexOf(" ");
	// 			if(i>0) ids.add(line.substring(0,i));
	// 			else ids.add(line);
	// 			line = in.readLine();
	// 		}
	// 	}
	// 	return ids;
		
	// }

    // @PostMapping("/readCsvFile")
    // public ResponseEntity<Map<String,String>> readCsvFile(MultipartFile file) {
    //     Map<String,String> response = new HashMap<>();
        
    //     try {
    //         List<String> listCsv = loadIds(file);
    //         String content = String.join(", ", listCsv);

    //         System.out.println("Contenido del archivo CSV:");
    //         System.out.println(content);
            
    //         response.put("data", content);
    //         return ResponseEntity.ok(response);

    //     } catch (Exception e) {
    //         response.put("message", "EERROR");
    //         return ResponseEntity.status(HttpStatus.FORBIDDEN).body(response);
    //     }

    // }
    
}
